<template>
  <v-container fluid class="monitor-page">
    <v-card>
      <v-tabs v-model="activeTab" bg-color="primary" dark>
        <v-tab value="live">
          <v-icon start>mdi-video</v-icon>
          Live Monitoring
        </v-tab>
        <v-tab value="upload">
          <v-icon start>mdi-upload</v-icon>
          Upload File
        </v-tab>
      </v-tabs>

      <v-divider />

      <v-window v-model="activeTab" class="pt-4">
        <!-- LIVE TAB -->
        <v-window-item value="live">
          <v-row>
            <v-col cols="12" md="8">
              <v-card class="preview-card">
                <div class="preview-controls">
                  <v-select
                    v-model="store.selectedSource"
                    :items="availableCameras"
                    item-title="label"
                    item-value="deviceId"
                    label="Select Camera"
                    dense
                    outlined
                    hide-details
                    class="mr-2"
                  />

                  <v-btn :color="store.capturing ? 'error' : 'primary'" @click="toggleCapture">
                    {{ store.capturing ? 'Stop' : 'Start' }} Capture
                  </v-btn>
                </div>

                <div class="d-flex align-center mt-4">
                  <v-switch
                    v-model="store.autoResume"
                    label="Auto-resume capture"
                    color="primary"
                    hide-details
                    inset
                    class="mr-4"
                  />
                  <v-switch
                    v-model="store.keepCameraAlive"
                    label="Keep camera open in background"
                    color="secondary"
                    hide-details
                    inset
                  />
                </div>

                <div class="preview-box mt-4">
                  <!-- Video Element for Real-time Feed -->
                  <video
                    v-show="store.selectedSource"
                    ref="videoEl"
                    autoplay
                    playsinline
                    muted
                    class="preview-video"
                  ></video>

                  <div v-if="!store.selectedSource" class="preview-placeholder">
                    <v-icon size="64" color="grey">mdi-video-off</v-icon>
                    <p>No camera available or selected.</p>
                  </div>
                </div>
              </v-card>
            </v-col>

            <v-col cols="12" md="4">
              <v-card class="logs-card">
                <div class="d-flex justify-space-between align-center mb-2">
                  <h2 class="logs-title">Detection Events</h2>
                </div>

                <div v-if="store.logs.length > 0" class="log-panel-container">
                  <v-expansion-panels variant="accordion">
                    <v-expansion-panel
                      v-for="(event, index) in store.logs"
                      :key="index"
                      class="log-event-panel"
                    >
                      <v-expansion-panel-title :class="getEventTitleClass(event)">
                        <v-icon start size="small" class="mr-2">mdi-clock-outline</v-icon>
                        {{ event.timestamp }}
                        <v-spacer></v-spacer>
                        <v-chip
                          v-if="event.info && !event.isError"
                          size="x-small"
                          color="blue-grey"
                          variant="tonal"
                          class="ml-2"
                          >Info</v-chip
                        >
                        <v-chip
                          v-if="event.isError"
                          size="x-small"
                          color="error"
                          variant="flat"
                          class="ml-2"
                          >Error</v-chip
                        >
                        <v-chip
                          v-if="!event.info && !event.isError"
                          size="x-small"
                          color="success"
                          variant="tonal"
                          class="ml-2"
                        >
                          {{ event.detected_items.length }} Found
                        </v-chip>
                        <v-chip
                          v-if="!event.info && !event.isError && event.missing_items.length > 0"
                          size="x-small"
                          color="warning"
                          variant="tonal"
                          class="ml-1"
                        >
                          {{ event.missing_items.length }} Missing
                        </v-chip>
                      </v-expansion-panel-title>

                      <v-expansion-panel-text class="log-event-details">
                        <div v-if="event.info">
                          <p :class="{ 'text-error': event.isError }">{{ event.info }}</p>
                        </div>
                        <div v-else>
                          <div v-if="event.detected_items.length > 0">
                            <strong class="text-success">Detected Items:</strong>
                            <ul class="compact-list">
                              <li
                                v-for="(item, i) in event.detected_items"
                                :key="`det-${index}-${i}`"
                              >
                                <!-- Colored Dot for Identification -->
                                <span
                                  class="color-dot"
                                  :style="{ backgroundColor: item.color || '#32CD32' }"
                                ></span>
                                {{ item.label }} ({{ item.confidence }})
                              </li>
                            </ul>
                          </div>
                          <v-alert v-else type="info" variant="text" density="compact" class="mt-2">
                            No items detected.
                          </v-alert>

                          <div v-if="event.missing_items.length > 0" class="mt-2">
                            <strong class="text-warning">Missing Items:</strong>
                            <ul class="compact-list">
                              <li
                                v-for="(item, i) in event.missing_items"
                                :key="`mis-${index}-${i}`"
                              >
                                {{ item }}
                              </li>
                            </ul>
                          </div>
                          <v-alert
                            v-else
                            type="success"
                            variant="text"
                            density="compact"
                            class="mt-2"
                          >
                            All expected items found.
                          </v-alert>
                        </div>
                      </v-expansion-panel-text>
                    </v-expansion-panel>
                  </v-expansion-panels>
                </div>
                <v-alert v-else type="info" variant="tonal">
                  Start capturing to see detection events.
                </v-alert>
              </v-card>
            </v-col>
          </v-row>
        </v-window-item>

        <!-- UPLOAD TAB -->
        <v-window-item value="upload">
          <v-row>
            <v-col cols="12" md="8">
              <v-card class="preview-card">
                <v-file-input
                  v-model="uploadedFileRef"
                  label="Upload Image"
                  dense
                  outlined
                  hide-details
                  accept="image/*"
                  @change="handleFileUpload"
                  clearable
                />

                <div class="d-flex align-center mt-3 flex-wrap gap-2">
                  <v-btn
                    color="primary"
                    :disabled="!uploadedFile"
                    @click="processUploadedFile"
                    class="mr-4"
                  >
                    Run Inference
                  </v-btn>

                  <v-slider
                    v-model="store.zoomLevel"
                    label="Zoom"
                    min="1.0"
                    max="3.0"
                    step="0.1"
                    thumb-label
                    hide-details
                    class="grow mr-4"
                  ></v-slider>

                  <!-- Controls for Bounding Boxes -->
                  <div v-if="currentDetections.length > 0" class="d-flex align-center">
                    <v-switch
                      v-model="showBoxes"
                      label="Show Boxes"
                      color="secondary"
                      hide-details
                      density="compact"
                      class="mr-4"
                    ></v-switch>
                  </div>
                </div>

                <div class="preview-box mt-4">
                  <!-- Container for aligned images -->
                  <div
                    v-if="store.uploadPreviewSrc && activeTab === 'upload'"
                    class="image-container"
                  >
                    <!--
                         Composite Image (Original Crop + Boxes drawn dynamically)
                         We no longer need a separate overlay <img> because we draw everything onto `uploadPreviewSrc`
                    -->
                    <img :src="store.uploadPreviewSrc" class="base-image" alt="Processed View" />
                  </div>

                  <div v-if="!store.uploadPreviewSrc" class="preview-placeholder">
                    <v-icon size="64" color="grey">mdi-image-outline</v-icon>
                    <p>Upload an image to see a preview.</p>
                  </div>
                </div>
              </v-card>
            </v-col>

            <v-col cols="12" md="4">
              <v-card class="logs-card">
                <div class="d-flex justify-space-between align-center mb-2">
                  <h2 class="logs-title">Upload Results</h2>
                </div>

                <div v-if="uploadLogs.length > 0" class="log-panel-container">
                  <v-expansion-panels variant="accordion">
                    <v-expansion-panel
                      v-for="(log, index) in uploadLogs"
                      :key="'upload-' + index"
                      class="log-event-panel"
                    >
                      <v-expansion-panel-title :color="getUploadLogColor(log.type)">
                        <template v-if="log.type === 'detection'">
                          <span
                            class="color-dot mr-2"
                            :style="{ backgroundColor: log.content.color || '#32CD32' }"
                          ></span>
                          <strong>{{ log.content.label }}</strong>
                          <v-chip size="small" class="ml-2" color="primary" variant="tonal">
                            Conf: {{ log.content.confidence }}
                          </v-chip>
                        </template>
                        <template v-else-if="log.type === 'summary'">
                          <v-icon start size="small">mdi-text-box-check-outline</v-icon>
                          <strong>Summary</strong>
                          <v-spacer></v-spacer>
                          <v-chip size="x-small" color="success" variant="tonal" class="ml-2">
                            {{ log.detectedCount }} Found
                          </v-chip>
                          <v-chip
                            v-if="log.missing_items.length > 0"
                            size="x-small"
                            color="warning"
                            variant="tonal"
                            class="ml-1"
                          >
                            {{ log.missing_items.length }} Missing
                          </v-chip>
                        </template>
                        <template v-else-if="log.type === 'info'">
                          <span
                            ><v-icon start>mdi-information-outline</v-icon>{{ log.content }}</span
                          >
                        </template>
                        <template v-else-if="log.type === 'error'">
                          <v-icon start>mdi-alert-circle-outline</v-icon>Error
                        </template>
                      </v-expansion-panel-title>

                      <v-expansion-panel-text
                        v-if="log.type === 'summary'"
                        class="pt-2 log-event-details"
                      >
                        <div v-if="log.missing_items.length > 0">
                          <strong>Missing Items:</strong>
                          <ul class="compact-list">
                            <li
                              v-for="(item, i) in log.missing_items"
                              :key="`mis-up-${index}-${i}`"
                            >
                              {{ item }}
                            </li>
                          </ul>
                        </div>
                        <v-alert v-else type="success" variant="text" density="compact"
                          >All items found.</v-alert
                        >
                      </v-expansion-panel-text>
                    </v-expansion-panel>
                  </v-expansion-panels>
                </div>
                <v-alert v-else type="info" variant="tonal">
                  Upload an image and run inference.
                </v-alert>
              </v-card>
            </v-col>
          </v-row>
        </v-window-item>
      </v-window>
    </v-card>
  </v-container>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { onBeforeRouteLeave } from 'vue-router'
import { useMonitorStore } from '@/stores/monitorStore'

const store = useMonitorStore()
const videoEl = ref(null)
const uploadedFileRef = ref(null)
const uploadedFile = ref(null)
const activeTab = ref('live')
const uploadLogs = ref([])
const availableCameras = ref([])
const originalUploadSrc = ref(null)
const currentDetections = ref([]) // Stores detection results with global coordinates
const showBoxes = ref(true)
const PREVIEW_MAX_DIM = 600

let captureInterval = null
let cleanupDone = false

// --- DATE HELPERS ---

function getFormattedLocalTimestamp() {
  const now = new Date()
  return now.toLocaleString('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

// --- CAMERA LOGIC ---

async function loadAvailableCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices()
    availableCameras.value = devices
      .filter((d) => d.kind === 'videoinput')
      .map((d, i) => ({
        label: d.label || `Camera ${i + 1}`,
        deviceId: d.deviceId,
      }))

    if (availableCameras.value.length > 0 && !store.selectedSource) {
      store.selectedSource = availableCameras.value[0].deviceId
    }
  } catch (err) {
    console.error('Failed to enumerate cameras:', err)
  }
}

navigator.mediaDevices.addEventListener('devicechange', loadAvailableCameras)

async function startCamera() {
  if (!store.selectedSource) {
    store.addInfoLog('No camera selected.', true)
    return
  }

  try {
    if (store.stream && store.stream.active) {
      if (videoEl.value) {
        videoEl.value.srcObject = store.stream
        // Ensure it plays
        videoEl.value.play().catch((e) => console.error('Error playing persistent stream:', e))
      }
      return
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: { exact: store.selectedSource } },
    })

    store.setStream(stream)

    if (videoEl.value) {
      videoEl.value.srcObject = stream
    } else {
      stream.getTracks().forEach((t) => t.stop())
      store.setStream(null)
      store.addInfoLog('Video element missing during start.', true)
    }
  } catch (err) {
    console.error('Camera start error:', err)
    store.addInfoLog('Camera access denied or failed.', true)
  }
}

function stopCamera() {
  // If "Keep camera open" is ON, we only clear the video element's source (viewfinder).
  // The tracks in store.stream remain active.
  if (store.keepCameraAlive) {
    if (videoEl.value) {
      videoEl.value.srcObject = null
      videoEl.value.load()
    }
    return
  }

  // Otherwise, full stop.
  if (store.stream) {
    store.stream.getTracks().forEach((t) => t.stop())
    store.setStream(null)
  }
  if (videoEl.value) {
    videoEl.value.srcObject = null
  }
}

async function captureFrame() {
  if (!videoEl.value || !store.capturing || !store.stream) return

  const canvas = document.createElement('canvas')
  if (videoEl.value.videoWidth === 0) return

  canvas.width = videoEl.value.videoWidth
  canvas.height = videoEl.value.videoHeight

  const ctx = canvas.getContext('2d')
  ctx.drawImage(videoEl.value, 0, 0, canvas.width, canvas.height)

  const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.9))
  if (!blob) return

  const formData = new FormData()
  formData.append('file', blob, 'frame.jpg')
  formData.append('zoom', 1.0)

  try {
    const res = await fetch('http://127.0.0.1:8000/api/infer', {
      method: 'POST',
      body: formData,
    })
    const result = await res.json()
    if (!res.ok) throw new Error(result.detail || `HTTP ${res.status}`)

    const formattedTimestamp = getFormattedLocalTimestamp()

    store.addCaptureEventLog({
      timestamp: formattedTimestamp,
      detected_items: result.detected_items || [],
      missing_items: result.missing_items || [],
    })

    try {
      await store.archiveLogsToHistory(
        [
          {
            timestamp: formattedTimestamp,
            detected_items: result.detected_items || [],
            missing_items: result.missing_items || [],
          },
        ],
        new Date().toISOString(),
      )
    } catch (saveErr) {
      console.warn('Instant save failed:', saveErr)
    }
  } catch (err) {
    console.error('Inference error:', err)
    store.addInfoLog(`Inference failed: ${err.message}`, true)
  }
}

async function toggleCapture() {
  if (store.capturing) {
    store.capturing = false
    if (captureInterval) clearInterval(captureInterval)
    captureInterval = null
    stopCamera()
  } else {
    if (!store.selectedSource) {
      store.addInfoLog('Please select a camera.', true)
      return
    }
    await startCamera()
    store.capturing = true
    if (captureInterval) clearInterval(captureInterval)
    captureInterval = setInterval(captureFrame, 60000)
  }
}

// --- UPLOAD LOGIC ---
async function updateZoomPreview() {
  if (!originalUploadSrc.value) {
    store.uploadPreviewSrc = null
    return
  }

  const img = new Image()
  img.onload = () => {
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    const { width, height } = img
    const zoom = store.zoomLevel
    const min_dim = Math.min(width, height)
    const crop_size = Math.max(1, Math.min(min_dim, Math.floor(min_dim / zoom)))
    const center_x = width / 2
    const center_y = height / 2
    const sx = center_x - crop_size / 2
    const sy = center_y - crop_size / 2

    // Canvas dimensions (preview size)
    const dw = Math.min(crop_size, PREVIEW_MAX_DIM)
    const dh = Math.min(crop_size, PREVIEW_MAX_DIM)

    canvas.width = dw
    canvas.height = dh

    // 1. Draw the base image crop
    ctx.drawImage(img, sx, sy, crop_size, crop_size, 0, 0, dw, dh)

    // 2. Draw boxes if enabled and available
    if (showBoxes.value && currentDetections.value.length > 0) {
      const scale = dw / crop_size // ratio of canvas_pixels to original_image_pixels

      currentDetections.value.forEach((det) => {
        if (!det.box) return
        const [gx1, gy1, gx2, gy2] = det.box
        const color = det.color || '#32CD32'

        // Convert Global Coords to Canvas Coords
        // coord_canvas = (coord_global - crop_start) * scale
        const cx1 = (gx1 - sx) * scale
        const cy1 = (gy1 - sy) * scale
        const cx2 = (gx2 - sx) * scale
        const cy2 = (gy2 - sy) * scale

        // Check if box is visible in current crop
        // Simple intersection check
        if (cx2 < 0 || cy2 < 0 || cx1 > dw || cy1 > dh) return

        // Draw Box
        ctx.strokeStyle = color
        ctx.lineWidth = 3
        ctx.strokeRect(cx1, cy1, cx2 - cx1, cy2 - cy1)

        // Draw Label
        const labelText = det.label
        ctx.font = '12px Arial'
        const textMetrics = ctx.measureText(labelText)
        const textWidth = textMetrics.width
        const textHeight = 14

        // Position text
        let tx = cx1 + 4
        let ty = cy1 + 4
        if (tx + textWidth > dw) tx = dw - textWidth - 4
        if (ty + textHeight > dh) ty = dh - textHeight - 4

        // Label Background
        ctx.fillStyle = color
        ctx.fillRect(tx - 2, ty - 2, textWidth + 4, textHeight + 4)

        // Label Text
        ctx.fillStyle = 'black'
        ctx.fillText(labelText, tx, ty + 10)
      })
    }

    store.uploadPreviewSrc = canvas.toDataURL('image/jpeg', 0.9)
    img.onload = null
  }
  img.src = originalUploadSrc.value
}

async function handleFileUpload(event) {
  const file = event.target.files[0]
  uploadedFile.value = file

  if (originalUploadSrc.value) {
    URL.revokeObjectURL(originalUploadSrc.value)
    originalUploadSrc.value = null
  }

  currentDetections.value = [] // Reset detections
  showBoxes.value = true

  if (!file) {
    store.uploadPreviewSrc = null
    uploadLogs.value = []
    return
  }

  originalUploadSrc.value = URL.createObjectURL(file)
  uploadLogs.value = [{ type: 'info', content: 'File ready.' }]
  await updateZoomPreview()
}

async function processUploadedFile() {
  if (!uploadedFile.value) {
    uploadLogs.value = [{ type: 'error', content: 'No file selected.' }]
    return
  }

  const formData = new FormData()
  formData.append('file', uploadedFile.value)
  // IMPORTANT: We now send the CURRENT zoom level, but the backend returns global coordinates
  formData.append('zoom', store.zoomLevel)

  uploadLogs.value = [{ type: 'info', content: 'Processing...' }]

  try {
    const res = await fetch('http://127.0.0.1:8000/api/infer', {
      method: 'POST',
      body: formData,
    })
    const result = await res.json()
    if (!res.ok) throw new Error(result.detail || `HTTP ${res.status}`)

    // Store detections globally so we can redraw them on zoom
    if (result.detected_items) {
      currentDetections.value = result.detected_items
      showBoxes.value = true
      // Trigger redraw to show boxes immediately
      await updateZoomPreview()
    }

    const detections = result.detected_items || []
    const missing = result.missing_items || []

    const summaryLog = {
      type: 'summary',
      detectedCount: detections.length,
      missing_items: missing,
      timestamp: result.timestamp,
    }
    const detectionLogs = detections.map((d) => ({
      type: 'detection',
      content: d,
    }))

    uploadLogs.value = [summaryLog, ...detectionLogs]
  } catch (err) {
    console.error('Upload inference failed:', err.message)
    uploadLogs.value = [{ type: 'error', content: `Error: ${err.message}` }]
    currentDetections.value = []
  }
}

function getUploadLogColor(type) {
  if (type === 'summary') return 'info'
  if (type === 'error') return 'error'
  return undefined
}

function getEventTitleClass(event) {
  if (event.isError) return 'bg-error-lighten-4'
  if (event.info) return 'bg-blue-grey-lighten-5'
  if (event.missing_items?.length > 0) return 'bg-warning-lighten-5'
  return 'bg-success-lighten-5'
}

// --- WATCHERS ---
watch(
  () => store.zoomLevel,
  () => {
    if (activeTab.value === 'upload' && originalUploadSrc.value) {
      updateZoomPreview()
    }
  },
)

watch(showBoxes, () => {
  if (activeTab.value === 'upload' && originalUploadSrc.value) {
    updateZoomPreview()
  }
})

watch(activeTab, (newTab) => {
  if (newTab === 'upload' && originalUploadSrc.value && !store.uploadPreviewSrc) {
    updateZoomPreview()
  }
  if (newTab === 'live' && store.capturing) {
    startCamera()
  } else if (newTab === 'live' && store.keepCameraAlive && store.stream) {
    setTimeout(() => {
      if (videoEl.value && store.stream) {
        videoEl.value.srcObject = store.stream
        videoEl.value.play().catch((e) => console.error('Error playing persistent stream:', e))
      }
    }, 100)
  }
})

// --- LIFECYCLE ---
onMounted(async () => {
  await loadAvailableCameras()
  store.startDailyCheck()

  if (store.keepCameraAlive && store.stream) {
    if (videoEl.value) {
      videoEl.value.srcObject = store.stream
      videoEl.value.play().catch((e) => console.error('Error re-playing persistent stream:', e))
    }
  }

  // Resume capturing if state indicates we should be running
  if (store.capturing && store.keepCameraAlive) {
    if (captureInterval) clearInterval(captureInterval)
    captureInterval = setInterval(captureFrame, 60000)
  }
})

function cleanup() {
  if (!cleanupDone) {
    if (captureInterval) clearInterval(captureInterval)
    captureInterval = null
    store.stopDailyCheck()

    // Only stop the actual camera tracks if keepCameraAlive is FALSE
    stopCamera()

    // Only reset capturing state if NOT keeping camera alive
    if (!store.keepCameraAlive) {
      store.capturing = false
    }

    if (originalUploadSrc.value) {
      URL.revokeObjectURL(originalUploadSrc.value)
      originalUploadSrc.value = null
    }
    cleanupDone = true
  }
}

onUnmounted(cleanup)
onBeforeRouteLeave(cleanup)
</script>

<style scoped>
.monitor-page {
  padding: 20px;
  min-height: calc(100vh - 144px);
}
.preview-card {
  padding: 16px;
  display: flex;
  flex-direction: column;
}
.preview-box {
  flex: 1;
  min-height: 400px;
  background-color: rgb(var(--v-theme-surface));
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}
.image-container {
  position: relative;
  display: inline-block;
  max-width: 100%;
  max-height: 400px;
}
.base-image {
  display: block;
  max-width: 100%;
  max-height: 400px;
  width: auto;
  height: auto;
  border-radius: 8px;
  object-fit: contain;
}
.overlay-image {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  border-radius: 8px;
  object-fit: contain;
  pointer-events: none;
  transition: opacity 0.2s ease-in-out;
}

.preview-video {
  width: 100%;
  height: 100%;
  border-radius: 8px;
  object-fit: contain;
}
.preview-placeholder {
  text-align: center;
  color: rgba(var(--v-theme-on-surface), 0.7);
}
.preview-controls {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}
.logs-card {
  padding: 16px;
  height: 100%;
  display: flex;
  flex-direction: column;
  background-color: rgb(var(--v-theme-surface));
}
.log-panel-container {
  flex-grow: 1;
  overflow-y: auto;
  margin-top: 8px;
}
.logs-title {
  font-size: 1.1rem;
  font-weight: 600;
}
.log-event-panel {
  margin-bottom: 4px;
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 4px !important;
}
.v-expansion-panel-title {
  min-height: 40px !important;
  padding: 8px 16px;
  font-size: 0.85rem;
}
.v-expansion-panel-text__wrapper {
  padding: 8px 16px 12px;
}
.compact-list {
  list-style-type: none;
  padding-left: 10px;
  margin: 0;
  font-size: 0.8rem;
}
.gap-2 {
  gap: 8px;
}
.color-dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-right: 6px;
}
</style>
