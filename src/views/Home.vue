<template>
  <div class="homepage-wrapper">
    <div class="homepage-center">
      <v-row justify="center" align="stretch" class="homepage-row">
        <!-- System status indicator -->
        <v-col cols="12" md="4">
          <v-card class="homepage-card">
            <h2 class="homepage-subtitle">Quick Status</h2>
            <v-chip :color="isOnline ? 'success' : 'error'" size="large">
              {{ isOnline ? 'System Online' : 'System Offline' }}
            </v-chip>
          </v-card>
        </v-col>

        <!-- Current date and time display -->
        <v-col cols="12" md="4">
          <v-card class="homepage-card">
            <h2 class="homepage-subtitle">Datetime</h2>
            <div class="homepage-datetime">{{ formattedTime }}</div>
          </v-card>
        </v-col>
      </v-row>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'

const isOnline = ref(false)
const currentTime = ref(new Date())

let statusInterval, clockInterval

// Check if the backend API is reachable
async function checkConnection() {
  try {
    const res = await fetch('http://localhost:8000/api/system-usage')
    // Any response means the server is reachable, even error responses
    isOnline.value = res.ok || res.status >= 400
  } catch (e) {
    // Network errors (no response) indicate the server is down
    isOnline.value = false
  }
}

// Format current time in Indonesian locale
const formattedTime = computed(() =>
  currentTime.value.toLocaleString('id-ID', {
    dateStyle: 'full',
    timeStyle: 'medium',
  }),
)

// Start periodic checks and clock updates
onMounted(() => {
  checkConnection()
  // Check connection status every 5 seconds
  statusInterval = setInterval(checkConnection, 5000)
  // Update clock display every second
  clockInterval = setInterval(() => (currentTime.value = new Date()), 1000)
})

// Clean up intervals when component is destroyed
onUnmounted(() => {
  clearInterval(statusInterval)
  clearInterval(clockInterval)
})
</script>

<style scoped>
.homepage-wrapper {
  height: calc(80vh - 50px);
  display: flex;
  justify-content: center;
  align-items: center;
  overflow: hidden;
  padding: 0 16px;
  box-sizing: border-box;
}

.homepage-center {
  width: 100%;
  max-width: 800px;
}

.homepage-row {
  width: 100%;
  max-width: 800px;
}

.homepage-card {
  padding: 20px;
  min-height: 140px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
}

.homepage-subtitle {
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 12px;
}

.homepage-datetime {
  font-size: 1rem;
  font-weight: 500;
}
</style>
