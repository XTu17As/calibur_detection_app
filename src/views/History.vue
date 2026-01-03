<template>
  <v-container fluid class="history-page">
    <h1 class="mb-6">Riwayat Deteksi Produk</h1>
    <v-alert v-if="error" type="error" variant="tonal" class="mb-4">
      {{ error }}
    </v-alert>

    <v-progress-circular
      v-if="loading"
      indeterminate
      color="primary"
      class="ma-auto d-block my-10"
    />

    <v-row v-else-if="sessions.length > 0" class="history-layout">
      <!-- Year selection sidebar -->
      <v-col cols="3" class="year-panel">
        <v-list nav dense>
          <v-list-item
            v-for="(yearSessions, year) in groupedByYear"
            :key="year"
            :active="selectedYear === year"
            @click="selectYear(year)"
            class="year-item"
          >
            <template #prepend>
              <v-icon color="primary">mdi-calendar</v-icon>
            </template>
            <v-list-item-title>
              {{ year }}
            </v-list-item-title>
            <v-list-item-subtitle> {{ yearSessions.length }} sesi </v-list-item-subtitle>
          </v-list-item>
        </v-list>
      </v-col>

      <!-- Session details panel -->
      <v-col cols="9" class="sessions-panel">
        <div v-if="selectedYear">
          <h2 class="mb-4">Sesi di {{ selectedYear }}</h2>

          <v-expansion-panels variant="accordion">
            <v-expansion-panel v-for="(day, i) in groupedByDate(selectedYear)" :key="i">
              <v-expansion-panel-title>
                <strong>{{ day.date }}</strong>
              </v-expansion-panel-title>
              <v-expansion-panel-text>
                <v-list density="comfortable">
                  <v-list-item
                    v-for="(session, idx) in day.sessions"
                    :key="idx"
                    class="session-entry"
                    @click="openSession(session)"
                  >
                    <template #prepend>
                      <v-icon color="primary">mdi-clock-time-four-outline</v-icon>
                    </template>
                    <v-list-item-title>
                      {{ formatTimeOnly(session.date) }}
                    </v-list-item-title>
                    <v-list-item-subtitle>
                      Missing: {{ countMissing(session.logs) }}, Detected:
                      {{ countDetected(session.logs) }}
                    </v-list-item-subtitle>
                  </v-list-item>
                </v-list>
              </v-expansion-panel-text>
            </v-expansion-panel>
          </v-expansion-panels>
        </div>

        <div v-else class="no-year">
          <v-icon size="60" color="grey">mdi-calendar-month-outline</v-icon>
          <p class="mt-2">Pilih tahun untuk melihat sesi</p>
        </div>
      </v-col>
    </v-row>

    <v-alert v-else type="info" variant="tonal" class="mt-4">
      Tidak ada riwayat deteksi yang ditemukan.
    </v-alert>

    <!-- Session detail dialog -->
    <v-dialog v-model="showDialog" max-width="700">
      <v-card v-if="activeSession">
        <v-card-title class="d-flex justify-space-between align-center">
          <div>
            <h3>{{ formatDateOnly(activeSession.date) }}</h3>
            <p class="text-medium-emphasis">
              {{ formatTimeOnly(activeSession.date) }}
            </p>
          </div>
          <v-btn icon @click="showDialog = false">
            <v-icon>mdi-close</v-icon>
          </v-btn>
        </v-card-title>

        <v-card-text>
          <v-chip-group column class="mb-4">
            <v-chip color="error" variant="flat">
              Total Missing: {{ countMissing(activeSession.logs) }}
            </v-chip>
            <v-chip color="success" variant="flat">
              Total Detected: {{ countDetected(activeSession.logs) }}
            </v-chip>
          </v-chip-group>

          <v-expansion-panels variant="accordion">
            <v-expansion-panel
              v-for="(group, product) in groupedProducts(activeSession.logs)"
              :key="product"
            >
              <v-expansion-panel-title>
                {{ product }}
              </v-expansion-panel-title>

              <v-expansion-panel-text>
                <v-list density="compact">
                  <v-list-item v-for="(log, idx) in group" :key="idx">
                    <v-list-item-title>
                      {{ log.timestamp }}
                    </v-list-item-title>
                    <v-list-item-subtitle>
                      <span v-if="log.missing" class="text-error">Missing</span>
                      <span v-if="log.detected" class="text-success"
                        >Detected ({{ log.confidence }})</span
                      >
                    </v-list-item-subtitle>
                  </v-list-item>
                </v-list>
              </v-expansion-panel-text>
            </v-expansion-panel>
          </v-expansion-panels>
        </v-card-text>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { decryptData } from '@/utils/crypto'

const backendURL = 'http://127.0.0.1:8000'

// Component state
const sessions = ref([]) // Decrypted history sessions from the backend
const loading = ref(true)
const error = ref(null)
const showDialog = ref(false)
const activeSession = ref(null) // Currently viewed session in the dialog
const selectedYear = ref(null) // Year selected in the sidebar

let controller = null // AbortController for cancelling fetch requests

// Fetch and decrypt detection history from the backend
async function fetchHistory() {
  loading.value = true
  error.value = null
  controller = new AbortController()

  try {
    const res = await fetch(`${backendURL}/api/history`, {
      signal: controller.signal,
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)

    const data = await res.json()
    if (!Array.isArray(data)) throw new Error('Invalid history format')

    // Decrypt each session and filter out invalid entries
    const decryptedSessions = data
      .map((item) => {
        if (typeof item === 'string') {
          try {
            return decryptData(item)
          } catch (e) {
            console.warn('Failed to decrypt history item:', e)
            return null
          }
        }
        return item
      })
      .filter((item) => item && item.date && Array.isArray(item.logs))

    // Sort sessions by date, newest first
    sessions.value = decryptedSessions.sort((a, b) => new Date(b.date) - new Date(a.date))

    // Auto-select the most recent year
    if (sessions.value.length > 0) {
      selectedYear.value = new Date(sessions.value[0].date).getFullYear().toString()
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      console.error('Failed to load history:', err)
      error.value = 'Gagal memuat data riwayat. Data mungkin rusak atau terenkripsi.'
    }
  } finally {
    loading.value = false
  }
}

// Group sessions by year for the sidebar display
const groupedByYear = computed(() => {
  const years = {}
  for (const s of sessions.value) {
    const y = new Date(s.date).getFullYear().toString()
    if (!years[y]) years[y] = []
    years[y].push(s)
  }
  // Sort years in descending order
  return Object.fromEntries(Object.entries(years).sort((a, b) => b[0] - a[0]))
})

// Group sessions by date within a selected year
function groupedByDate(year) {
  const filtered = sessions.value.filter((s) => new Date(s.date).getFullYear().toString() === year)
  const groups = {}

  for (const s of filtered) {
    const d = formatDateOnly(s.date)
    if (!groups[d]) groups[d] = []
    groups[d].push(s)
  }

  return Object.entries(groups).map(([date, sessions]) => ({
    date,
    sessions: sessions.sort((a, b) => new Date(b.date) - new Date(a.date)),
  }))
}

// Update the selected year in the sidebar
function selectYear(year) {
  selectedYear.value = year
}

// Open a session detail dialog
function openSession(session) {
  activeSession.value = session
  showDialog.value = true
}

// Format date as full date string in Indonesian locale
function formatDateOnly(isoDate) {
  const date = new Date(isoDate)
  return date.toLocaleString('id-ID', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })
}

// Format date as time string in Indonesian locale
function formatTimeOnly(isoDate) {
  const date = new Date(isoDate)
  return date.toLocaleTimeString('id-ID', {
    hour: '2-digit',
    minute: '2-digit',
  })
}

// Count total missing items across all logs in a session
function countMissing(logs) {
  return logs.reduce((total, log) => {
    if (log && log.missing_items) {
      total += log.missing_items.length
    }
    return total
  }, 0)
}

// Count total detected items across all logs in a session
function countDetected(logs) {
  return logs.reduce((total, log) => {
    if (log && log.detected_items) {
      total += log.detected_items.length
    }
    return total
  }, 0)
}

// Group logs by product name for detailed session view
// Flattens detected and missing items into per-product timelines
function groupedProducts(logs) {
  const products = {}
  // Filter out info/error logs to focus on detection events
  const detectionLogs = logs.filter((log) => log && !log.info && !log.isError)

  for (const log of detectionLogs) {
    // Process detected items
    if (log.detected_items) {
      for (const item of log.detected_items) {
        const productName = item.label
        if (!products[productName]) {
          products[productName] = []
        }
        products[productName].push({
          timestamp: log.timestamp,
          detected: true,
          confidence: item.confidence,
        })
      }
    }

    // Process missing items
    if (log.missing_items) {
      for (const productName of log.missing_items) {
        if (!products[productName]) {
          products[productName] = []
        }
        products[productName].push({
          timestamp: log.timestamp,
          missing: true,
        })
      }
    }
  }

  // Sort each product's timeline by timestamp, newest first
  for (const productName in products) {
    products[productName].sort((a, b) => {
      // Parse timestamp format "DD MMM YYYY, HH:MM:SS"
      const dateA = new Date(a.timestamp.replace(/(\d{2}) (\w{3}) (\d{4}),/, '$2 $1 $3'))
      const dateB = new Date(b.timestamp.replace(/(\d{2}) (\w{3}) (\d{4}),/, '$2 $1 $3'))
      return dateB - dateA
    })
  }

  return products
}

// Load history data when component mounts
onMounted(fetchHistory)

// Cancel ongoing fetch requests when component unmounts
onBeforeUnmount(() => {
  if (controller) controller.abort()
})
</script>

<style scoped>
.history-page {
  padding: 24px;
  min-height: calc(100vh - 144px);
}

.year-panel {
  border-right: 1px solid rgba(0, 0, 0, 0.08);
  height: calc(100vh - 200px);
  overflow-y: auto;
}

.year-item {
  border-radius: 8px;
  margin-bottom: 4px;
}

.year-item.v-list-item--active {
  background-color: rgba(var(--v-theme-primary), 0.12);
}

.sessions-panel {
  padding-left: 16px;
  height: calc(100vh - 210px);
  overflow-y: auto;
}

.session-entry {
  cursor: pointer;
  border-bottom: 1px solid rgba(0, 0, 0, 0.06);
}

.no-year {
  text-align: center;
  margin-top: 120px;
  color: grey;
}

.v-dialog .v-card {
  border-radius: 12px;
}

.text-error {
  color: rgb(var(--v-theme-error));
  font-weight: 500;
}

.text-success {
  color: rgb(var(--v-theme-success));
}
</style>
