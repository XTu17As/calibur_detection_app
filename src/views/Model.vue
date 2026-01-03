<template>
  <v-container fluid class="model-page">
    <h1 class="mb-6">Manajemen Model</h1>
    <v-alert v-if="error" type="error" variant="tonal" class="mb-4">
      {{ error }}
    </v-alert>

    <v-progress-circular
      v-if="loading"
      indeterminate
      color="primary"
      class="ma-auto d-block my-10"
    />

    <v-row v-else-if="groupedModels.length > 0" class="model-layout">
      <!-- Left sidebar: shows all available models organized by training iteration -->
      <v-col cols="3" class="model-list-panel">
        <v-list nav dense>
          <v-list-group
            v-for="group in groupedModels"
            :key="group.iteration"
            :value="group.iteration"
          >
            <template v-slot:activator="{ props }">
              <v-list-item
                v-bind="props"
                :title="group.iteration"
                class="model-folder-item font-weight-bold text-primary"
                prepend-icon="mdi-folder"
              ></v-list-item>
            </template>

            <v-btn
              v-for="model in group.models"
              :key="model.name"
              :variant="selectedModel === model.name ? 'flat' : 'tonal'"
              :color="
                model.is_active ? 'success' : selectedModel === model.name ? 'primary' : 'grey'
              "
              class="mb-2 justify-start text-left ml-16"
              style="width: 16rem"
              @click="selectedModel = model.name"
              rounded="lg"
            >
              <v-icon start :color="model.is_active ? 'white' : 'grey-lighten-3'" size="small">
                {{ model.is_active ? 'mdi-check-circle' : 'mdi-brain' }}
              </v-icon>
              {{ model.shortName }}
              <v-chip
                v-if="model.is_active"
                color="success"
                size="x-small"
                class="ml-auto text-white"
              >
                Aktif
              </v-chip>
            </v-btn>
          </v-list-group>
        </v-list>
      </v-col>

      <!-- Right panel: shows details and specs for the selected model -->
      <v-col cols="9" class="model-details-panel">
        <div v-if="selectedModel">
          <h2 class="mb-4">Spesifikasi {{ selectedModelShortName }}</h2>
          <v-divider class="my-4" />

          <v-list density="compact" v-if="currentSpecs.length > 0">
            <v-list-item v-for="(spec, i) in currentSpecs" :key="i">
              <v-list-item-title>{{ spec.label }}</v-list-item-title>
              <v-list-item-subtitle>{{ spec.value }}</v-list-item-subtitle>
            </v-list-item>
          </v-list>

          <!-- Display detailed error messages if model loading fails -->
          <v-alert
            v-if="applyError"
            type="error"
            variant="tonal"
            class="my-4"
            style="
              white-space: pre-wrap;
              font-family: monospace;
              font-size: 0.8rem;
              max-height: 300px;
              overflow-y: auto;
            "
          >
            <strong>Error Applying Model:</strong>
            <div class="mt-2">{{ applyError }}</div>
          </v-alert>

          <v-divider class="my-4" />

          <v-btn
            :color="isModelActive ? 'success' : 'primary'"
            @click="applyModel"
            :disabled="!selectedModel || isApplying || isModelActive"
            :loading="isApplying"
          >
            <v-icon start>mdi-check</v-icon>
            {{ isModelActive ? 'Applied' : 'Apply' }}
          </v-btn>
        </div>

        <div v-else class="no-model-selected">
          <v-icon size="60" color="grey">mdi-brain</v-icon>
          <p class="mt-2">Pilih model untuk melihat detail & menerapkan</p>
        </div>
      </v-col>
    </v-row>

    <v-alert v-else type="info" variant="tonal" class="mt-4">
      Tidak ada model deteksi yang ditemukan.
    </v-alert>

    <v-snackbar v-model="snackbar.show" :color="snackbar.color" timeout="2500" top elevation="8">
      {{ snackbar.message }}
      <template #actions>
        <v-btn icon @click="snackbar.show = false">
          <v-icon>mdi-close</v-icon>
        </v-btn>
      </template>
    </v-snackbar>
  </v-container>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'

// Component state management
const models = ref([]) // Full list of available models from the API
const selectedModel = ref(null) // Currently selected model path (e.g., "legacy/model1")
const loading = ref(true)
const error = ref(null) // Tracks errors during initial model fetch
const applyError = ref(null) // Tracks errors when applying a model
const isApplying = ref(false) // Loading state for the apply button

// Snackbar notification state
const snackbar = ref({
  show: false,
  message: '',
  color: 'success',
})

// Group models by their training iteration folder
// Converts flat model list into nested structure for sidebar display
const groupedModels = computed(() => {
  const groups = new Map()

  for (const model of models.value) {
    const parts = model.name.split('/')
    const iteration = parts[0] || 'unknown'
    const shortName = parts[1] || model.name

    if (!groups.has(iteration)) {
      groups.set(iteration, [])
    }

    groups.get(iteration).push({
      ...model,
      shortName: shortName,
    })
  }

  // Sort both iterations and models alphabetically for consistent display
  return Array.from(groups.entries())
    .map(([iteration, models]) => ({
      iteration: iteration,
      models: models.sort((a, b) => a.shortName.localeCompare(b.shortName)),
    }))
    .sort((a, b) => a.iteration.localeCompare(b.iteration))
})

// Extract specifications for the currently selected model
const currentSpecs = computed(() => {
  const found = models.value.find((m) => m.name === selectedModel.value)
  return found ? found.specs : []
})

// Get display name for selected model (without folder prefix)
const selectedModelShortName = computed(() => {
  if (!selectedModel.value) return 'Model'
  const parts = selectedModel.value.split('/')
  return parts[1] || parts[0]
})

// Check if the selected model is already active on the backend
const isModelActive = computed(() => {
  if (!selectedModel.value || !models.value.length) return false
  const found = models.value.find((m) => m.name === selectedModel.value)
  return found ? found.is_active : false
})

// Fetch available models from the backend API
async function fetchModels() {
  loading.value = true
  error.value = null

  try {
    const res = await fetch('http://127.0.0.1:8000/api/models')

    if (!res.ok) {
      throw new Error(`Backend returned status ${res.status}`)
    }

    const data = await res.json()
    models.value = data

    // Auto-select the currently active model if one exists
    const active = data.find((m) => m.is_active)
    selectedModel.value = active ? active.name : null
  } catch (err) {
    console.error('Failed to fetch models:', err)
    error.value = 'Gagal memuat daftar model. Backend mungkin offline.'
    snackbar.value = {
      show: true,
      message: 'Gagal memuat daftar model.',
      color: 'error',
    }
  } finally {
    loading.value = false
  }
}

// Send request to backend to load the selected model into memory
// This tells the inference server to load the model weights
async function applyModel() {
  if (!selectedModel.value) return

  applyError.value = null
  isApplying.value = true

  try {
    const formData = new FormData()
    formData.append('model_name', selectedModel.value)

    const res = await fetch('http://127.0.0.1:8000/api/models/load', {
      method: 'POST',
      body: formData,
    })

    const result = await res.json()

    if (!res.ok) {
      throw new Error(result.detail || `Model load failed with status ${res.status}`)
    }

    snackbar.value = {
      show: true,
      message: result.message || `Model "${selectedModelShortName.value}" diterapkan.`,
      color: 'success',
    }

    // Refresh the model list to update active status indicators
    await fetchModels()
  } catch (err) {
    console.error('Error applying model:', err.message)
    applyError.value = err.message
    snackbar.value = {
      show: true,
      message: 'Gagal menerapkan model. Lihat detail error.',
      color: 'error',
    }
  } finally {
    isApplying.value = false
  }
}

// Load available models when component mounts
onMounted(fetchModels)
</script>

<style scoped>
.model-page {
  padding: 24px;
  min-height: calc(100vh - 144px);
}

.model-list-panel {
  border-right: 1px solid rgba(0, 0, 0, 0.08);
  height: calc(100vh - 210px);
  overflow-y: auto;
}

.model-folder-item {
  border-radius: 8px;
  margin-bottom: 4px;
}

.model-folder-item.v-list-item--active {
  background-color: rgba(var(--v-theme-primary), 0.12);
}

.model-details-panel {
  padding-left: 16px;
}

.no-model-selected {
  text-align: center;
  margin-top: 120px;
  color: grey;
}

.v-list-item-title {
  font-weight: 500;
}

.v-list-item-subtitle {
  opacity: 0.8;
}

.v-btn.ml-4 {
  width: calc(100% - 16px);
  height: 36px !important;
}
</style>
