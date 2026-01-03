import { defineStore } from 'pinia'
import { ref } from 'vue'
import { encryptData, decryptData } from '@/utils/crypto'

export const useMonitorStore = defineStore('monitor', () => {
  // Detection event logs displayed in the UI
  const logs = ref([])
  // All logs captured during the current day's session
  const todaySessionLogs = ref([])
  // Historical data from previous sessions
  const history = ref([])

  // Camera and capture state
  const capturing = ref(false)
  const selectedSource = ref(null)
  const stream = ref(null) // Active MediaStream from getUserMedia

  // UI state and user preferences
  const livePreviewSrc = ref(null)
  const uploadPreviewSrc = ref(null)
  const autoResume = ref(false) // Whether to resume capture after interruption
  const keepCameraAlive = ref(false) // Maintain stream when not actively capturing
  const zoomLevel = ref(1.0)
  const uploadColorLegend = ref({})

  // LocalStorage keys for data persistence
  const LOG_KEY = 'monitorUiLogs'
  const SESSION_LOG_KEY = 'monitorSessionLogs'
  const LAST_DAY_KEY = 'monitorLastDay'

  const dayCheckInterval = ref(null)

  // Save current logs to browser storage
  function persistLogs() {
    const today = new Date().toISOString().split('T')[0]
    localStorage.setItem(LOG_KEY, JSON.stringify(logs.value))
    localStorage.setItem(SESSION_LOG_KEY, JSON.stringify(todaySessionLogs.value))
    localStorage.setItem(LAST_DAY_KEY, today)
  }

  // Load logs from storage and handle day transitions
  async function loadLogs() {
    try {
      const savedDay = localStorage.getItem(LAST_DAY_KEY)
      const today = new Date().toISOString().split('T')[0]

      // Archive previous day's logs when a new day starts
      if (savedDay && savedDay !== today) {
        console.log('New day detected. Archiving old logs.')
        const oldSessionLogs = JSON.parse(localStorage.getItem(SESSION_LOG_KEY) || '[]')

        if (oldSessionLogs.length > 0) {
          await archiveLogsToHistory(oldSessionLogs, savedDay)
        }

        // Reset daily logs and update the stored date
        clearLogs()
        localStorage.setItem(LAST_DAY_KEY, today)
      } else {
        // Load existing logs for the current day
        const savedLogs = JSON.parse(localStorage.getItem(LOG_KEY) || '[]')
        const savedSessionLogs = JSON.parse(localStorage.getItem(SESSION_LOG_KEY) || '[]')
        logs.value = Array.isArray(savedLogs) ? savedLogs : []
        todaySessionLogs.value = Array.isArray(savedSessionLogs) ? savedSessionLogs : []
        if (!savedDay) localStorage.setItem(LAST_DAY_KEY, today)
      }
    } catch (err) {
      console.error('Error loading logs:', err)
      clearLogs()
      localStorage.setItem(LAST_DAY_KEY, new Date().toISOString().split('T')[0])
    }
  }

  // Add a detection event to the log
  // Accepts either an API response object or a simple string message
  function addCaptureEventLog(captureResult) {
    let eventEntry

    if (typeof captureResult === 'string') {
      eventEntry = {
        timestamp: new Date().toLocaleString('en-GB'),
        detected_items: [],
        missing_items: [],
        info: captureResult,
      }
    } else if (captureResult && captureResult.timestamp) {
      eventEntry = {
        timestamp: captureResult.timestamp,
        detected_items: captureResult.detected_items || [],
        missing_items: captureResult.missing_items || [],
      }
    } else {
      console.error('Invalid capture result format:', captureResult)
      return
    }

    logs.value.unshift(eventEntry)
    // Keep only the most recent 50 logs in memory
    if (logs.value.length > 50) logs.value.pop()

    todaySessionLogs.value.unshift(eventEntry)
    persistLogs()
  }

  // Add an informational or error message to the log
  function addInfoLog(message, isError = false) {
    const eventEntry = {
      timestamp: new Date().toLocaleString('en-GB'),
      detected_items: [],
      missing_items: [],
      info: message,
      isError,
    }

    logs.value.unshift(eventEntry)
    if (logs.value.length > 50) logs.value.pop()
    todaySessionLogs.value.unshift(eventEntry)
    persistLogs()
  }

  // Remove all logs from memory and storage
  function clearLogs() {
    logs.value = []
    todaySessionLogs.value = []
    localStorage.removeItem(LOG_KEY)
    localStorage.removeItem(SESSION_LOG_KEY)
  }

  // Send logs to the backend for long-term storage
  // Encrypts the data before transmission
  async function archiveLogsToHistory(logsToSave, dateString) {
    if (!logsToSave || logsToSave.length === 0) {
      return
    }

    try {
      const session = {
        date: new Date(dateString).toISOString(),
        logs: logsToSave,
      }

      const encrypted = encryptData(session)

      // Fetch existing history
      const res = await fetch('http://127.0.0.1:8000/api/history')
      if (!res.ok) throw new Error(`Failed to fetch history: ${res.status}`)

      let existing = await res.json()
      if (!Array.isArray(existing)) existing = []

      // Append new session to history
      const updated = [...existing, encrypted]

      await fetch('http://127.0.0.1:8000/api/history', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updated),
      })

      console.log(`Session archived for ${dateString}`)
    } catch (err) {
      console.error('archiveLogsToHistory failed:', err)
      addInfoLog(`Failed to archive logs for ${dateString}: ${err.message}`, true)
    }
  }

  // Release all camera resources and stop the stream
  // This is a hard stop used during cleanup or navigation away
  function stopAllCameras() {
    if (stream.value) {
      stream.value.getTracks().forEach((t) => t.stop())
      stream.value = null
    }

    // Fallback: manually find and stop any active video element streams
    const video = document.querySelector('video')
    if (video && video.srcObject) {
      const tracks = video.srcObject.getTracks()
      tracks.forEach((t) => t.stop())
      video.srcObject = null
    }

    if (capturing.value) capturing.value = false
  }

  // Update the active camera stream reference
  function setStream(newStream) {
    stream.value = newStream
  }

  // Start monitoring for day changes to trigger log archival
  // Runs every 5 minutes to check if the date has changed
  async function startDailyCheck() {
    if (dayCheckInterval.value) return
    await loadLogs()
    dayCheckInterval.value = setInterval(async () => {
      const savedDay = localStorage.getItem(LAST_DAY_KEY)
      const today = new Date().toISOString().split('T')[0]
      if (savedDay && savedDay !== today) {
        console.log('Interval check found new day.')
        await loadLogs()
      }
    }, 300000) // Check every 5 minutes
  }

  // Stop the daily check interval
  function stopDailyCheck() {
    if (dayCheckInterval.value) {
      clearInterval(dayCheckInterval.value)
      dayCheckInterval.value = null
    }
  }

  return {
    logs,
    history,
    capturing,
    selectedSource,
    stream,
    livePreviewSrc,
    uploadPreviewSrc,
    autoResume,
    keepCameraAlive,
    zoomLevel,
    uploadColorLegend,
    addCaptureEventLog,
    addInfoLog,
    clearLogs,
    stopAllCameras,
    setStream,
    loadLogs,
    startDailyCheck,
    stopDailyCheck,
    archiveLogsToHistory,
  }
})
