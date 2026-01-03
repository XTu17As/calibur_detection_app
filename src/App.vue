<template>
  <v-app :class="appBgClass">
    <v-app-bar app flat :color="isDark ? 'grey-darken-4' : 'white'">
      <v-toolbar-title class="font-weight-medium" :class="isDark ? 'text-white' : 'text-black'">
        CALIBUR
      </v-toolbar-title>

      <v-spacer />
      <v-btn
        v-for="item in navItems"
        :key="item.title"
        :to="item.route"
        variant="text"
        :class="isDark ? 'text-grey-lighten-2' : 'text-grey-darken-2'"
      >
        <v-icon start>{{ item.icon }}</v-icon>
        {{ item.title }}
      </v-btn>

      <v-divider vertical style="padding-left: 1rem" />
      <v-btn
        variant="text"
        @click="toggleTheme"
        :icon="isDark ? 'mdi-weather-sunny' : 'mdi-weather-night'"
        :color="isDark ? 'white' : 'black'"
      />
    </v-app-bar>

    <v-main>
      <router-view />
    </v-main>

    <v-footer
      app
      height="auto"
      :color="isDark ? 'grey-darken-4' : 'white'"
      class="px-6 py-2 d-flex align-center flex-wrap"
      style="gap: 16px; min-height: 80px"
    >
      <div
        class="footer-usage-row d-flex align-center"
        style="gap: 24px; flex: 1 1 auto; justify-content: flex-start"
      >
        <div class="usage-column">
          <div class="usage-item">
            <span class="footer-label">CPU: {{ usage.cpu }}%</span>
            <v-progress-linear :model-value="usage.cpu" :color="cpuColor" height="10" rounded />
          </div>

          <div class="usage-item">
            <span class="footer-label">Memory: {{ usage.memory }}%</span>
            <v-progress-linear
              :model-value="usage.memory"
              :color="memoryColor"
              height="10"
              rounded
            />
          </div>
        </div>

        <div class="usage-column" v-if="usage.gpu !== null || usage.vram !== null">
          <div v-if="usage.gpu !== null" class="usage-item">
            <span class="footer-label">GPU: {{ usage.gpu }}%</span>
            <v-progress-linear :model-value="usage.gpu" :color="gpuColor" height="10" rounded />
          </div>
          <div v-if="usage.vram !== null" class="usage-item">
            <span class="footer-label">VRAM: {{ usage.vram }}%</span>
            <v-progress-linear :model-value="usage.vram" :color="vramColor" height="10" rounded />
          </div>
        </div>

        <div class="usage-column" v-if="usage.gpu === null && usage.vram === null">
          <div class="usage-item">
            <span class="footer-label">GPU:</span>
            <div class="gpu-label">No GPU detected</div>
          </div>
        </div>
      </div>

      <div
        class="footer-model-status text-center"
        style="flex: 0 1 auto; transform: translate(-20%, -20%)"
      >
        <span class="footer-label">Active Model</span>
        <v-chip v-if="activeModelName" color="primary" size="small" variant="flat" class="mt-1">
          <v-icon start>mdi-brain</v-icon>
          {{ activeModelName }}
        </v-chip>
        <v-chip v-else color="grey" size="small" variant="tonal" class="mt-1">
          <v-icon start>mdi-alert-circle-outline</v-icon>
          No Model Loaded
        </v-chip>
      </div>

      <div
        class="footer-time d-flex align-center"
        style="flex: 1 1 auto; justify-content: flex-end"
      >
        <v-icon class="mr-2" :color="isDark ? 'grey-lighten-1' : 'grey-darken-3'"
          >mdi-clock-outline</v-icon
        >
        <div
          class="footer-time-text mr-4"
          :class="isDark ? 'text-grey-lighten-1' : 'text-grey-darken-3'"
        >
          <div class="time-top">{{ formattedTime }}</div>
          <div class="time-bottom">{{ formattedDate }}</div>
        </div>
        <v-chip :color="isOnline ? 'success' : 'error'" size="small">
          {{ isOnline ? 'Online' : 'Offline' }}
        </v-chip>
      </div>
    </v-footer>
  </v-app>
</template>

<script setup>
// All the imports. If you remove one, it all breaks. Probably.
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useTheme } from 'vuetify'

// Get the theme object. This is how we do the dark mode thing.
const theme = useTheme()
// 'isDark' is a computed property, which is Vue's fancy way of saying
// "figure this out whenever it changes, I'm too busy."
const isDark = computed(() => theme.global.name.value === 'dark')

// Toggles the theme. Groundbreaking.
function toggleTheme() {
  theme.global.name.value = isDark.value ? 'light' : 'dark'
  // Shove it in localStorage so the user doesn't blind themselves on refresh.
  localStorage.setItem('theme', theme.global.name.value)
}

// When this thing finally mounts...
onMounted(() => {
  // Check if the user *already* saved themselves from light mode.
  const saved = localStorage.getItem('theme')
  if (saved) {
    theme.global.name.value = saved // Good. Stay in the dark.
  }
})

// All the pages. Don't ask me to add another one.
const navItems = [
  { title: 'Home', icon: 'mdi-home', route: '/' },
  { title: 'Monitor', icon: 'mdi-monitor-dashboard', route: '/monitor' },
  { title: 'Model', icon: 'mdi-brain', route: '/model' },
  { title: 'Riwayat', icon: 'mdi-history', route: '/history' },
  { title: 'Help', icon: 'mdi-help-circle-outline', route: '/help' },
  { title: 'About', icon: 'mdi-information-outline', route: '/about' },
]

// The default state for our panic dashboard.
const usage = ref({
  cpu: 0,
  memory: 0,
  gpu: null, // Null means we're probably on a server... or a toaster.
  vram: null,
})

// State. Just... state.
const currentTime = ref(new Date())
const isOnline = ref(false) // Assume the worst.
const activeModelName = ref(null) // NEW: To store the active model name
let usageInterval, clockInterval, statusInterval, modelInterval // Globals. I know, I know. Sue me.

// Goes and fetches the usage stats from the backend.
async function fetchUsage() {
  try {
    // Fire and forget. If it fails, the user just sees 0%. Ignorance is bliss.
    const res = await fetch('http://localhost:8000/api/system-usage')
    usage.value = await res.json()
  } catch (e) {
    // Yeah, it failed. What a surprise.
    console.error('Failed to fetch usage. The backend is probably on fire.', e)
    // Don't set usage to 0 here, it'll just flicker. Let it be wrong.
  }
}

// NEW: Fetches the currently loaded model name
async function fetchActiveModel() {
  try {
    const res = await fetch('http://localhost:8000/api/models/current')
    if (!res.ok) throw new Error('Failed to get model status')
    const data = await res.json()
    activeModelName.value = data.current_model // e.g., "legacy/model_v1" or null
  } catch (e) {
    console.error('Failed to fetch active model:', e)
    activeModelName.value = null // Assume no model if fetch fails
  }
}

// Pings the backend to see if the API is *actually* online.
async function checkConnection() {
  try {
    const res = await fetch('http://localhost:8000/api/system-usage')
    // If we get a response (even a 4xx or 5xx), the server is reachable.
    // We'll consider it "Online" if the fetch itself doesn't throw an error.
    if (res.ok) {
      isOnline.value = true // We live!
    } else {
      // The server responded with an error, but it's still "online".
      isOnline.value = true
    }
  } catch (e) {
    isOnline.value = false // We're dead, Jim.
  }
}

// Makes the clock look like a clock.
const formattedTime = computed(() => {
  const d = currentTime.value
  // All this just to add a leading zero.
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(
    2,
    '0',
  )}:${String(d.getSeconds()).padStart(2, '0')}`
})

// Makes the date look like a date.
const formattedDate = computed(() => {
  const d = currentTime.value
  // Why are arrays zero-indexed. WHY.
  const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
  return `${days[d.getDay()]}, ${String(d.getDate()).padStart(
    2,
    '0',
  )}-${String(d.getMonth() + 1).padStart(2, '0')}-${d.getFullYear()}`
})

// These just change the color of the bars from "fine" to "ON FIRE".
// UPDATED: Now checks if a model is active first.
const neutralColor = computed(() => (isDark.value ? 'grey-darken-1' : 'grey-lighten-1'))

const cpuColor = computed(() => {
  if (!activeModelName.value) return neutralColor.value // No model, neutral color
  return usage.value.cpu < 50 ? 'green' : usage.value.cpu < 80 ? 'orange' : 'red'
})
const memoryColor = computed(() => {
  if (!activeModelName.value) return neutralColor.value
  return usage.value.memory < 50 ? 'green' : usage.value.memory < 80 ? 'orange' : 'red'
})
const gpuColor = computed(() => {
  if (!activeModelName.value) return neutralColor.value
  return usage.value.gpu === null
    ? 'grey'
    : usage.value.gpu < 50
      ? 'green'
      : usage.value.gpu < 80
        ? 'orange'
        : 'red'
})
const vramColor = computed(() => {
  if (!activeModelName.value) return neutralColor.value
  return usage.value.vram === null
    ? 'grey'
    : usage.value.vram < 50
      ? 'green'
      : usage.value.vram < 80
        ? 'orange'
        : 'red'
})

// Kick off all the intervals. This is where the magic (and the memory leaks) happen.
onMounted(() => {
  fetchUsage() // Run once immediately, so it's not 0.
  checkConnection() // Same here.
  fetchActiveModel() // NEW: Run this once too.

  // Now do it forever.
  usageInterval = setInterval(fetchUsage, 3000) // 3 seconds is fine, right?
  statusInterval = setInterval(checkConnection, 5000) // 5 seconds.
  clockInterval = setInterval(() => (currentTime.value = new Date()), 1000) // 1 second.
  modelInterval = setInterval(fetchActiveModel, 5000) // NEW: Check model status every 5s
})

// The "cleanup" function. This is supposed to run.
// If it doesn't, enjoy the 50 intervals running in the background.
onUnmounted(() => {
  clearInterval(usageInterval)
  clearInterval(statusInterval)
  clearInterval(clockInterval)
  clearInterval(modelInterval) // NEW: Clean up the model interval
})

// This just sets the background color. Why is it a computed property?
// ...because `isDark` is. Stop asking questions.
const appBgClass = computed(() =>
  isDark.value ? 'bg-grey-darken-4 text-white' : 'bg-grey-lighten-5 text-black',
)
</script>

<style scoped>
/* I hate CSS. */
.footer-label {
  font-size: 0.75rem;
  display: block;
  margin-bottom: 2px;
}
.footer-usage-row {
  display: flex;
  gap: 16px;
  /* max-width: 800px; <-- REMOVED this to allow better wrapping */
}
/* NEW: Column for stacking usage bars */
.usage-column {
  display: flex;
  flex-direction: column;
  gap: 4px; /* Space between CPU and Mem */
}
.usage-item {
  width: 130px;
}
.gpu-label {
  font-size: 0.8rem;
  font-weight: 500;
  margin-top: 2px;
}
.footer-time-text {
  display: flex;
  flex-direction: column;
  line-height: 1.2;
}
.time-top {
  font-size: 1rem;
  font-weight: 600;
}
.time-bottom {
  font-size: 0.8rem;
  margin-top: 2px;
}
</style>
