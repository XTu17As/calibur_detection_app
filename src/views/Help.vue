<!-- Oh god help me -->
<template>
  <v-container fluid class="help-page">
    <h1 class="mb-6">Panduan & Bantuan</h1>
    <!-- "Help". Because "RTFM" isn't professional. -->

    <v-row>
      <!-- LEFT: FAQ List -->
      <!-- A list of questions we're already tired of. -->
      <v-col cols="12" md="4" class="faq-list">
        <v-card class="pa-4">
          <v-card-title class="text-h6 font-weight-bold">
            <v-icon start color="primary">mdi-help-circle-outline</v-icon>
            FAQ
          </v-card-title>
          <v-divider class="my-2" />

          <v-list nav density="comfortable">
            <!-- Loop through the static, hardcoded questions. -->
            <v-list-item
              v-for="(q, i) in faqs"
              :key="i"
              :active="selected === i"
              rounded
              class="faq-item"
              @click="selected = i"
            >
              <template #prepend>
                <v-icon color="primary">mdi-comment-question-outline</v-icon>
              </template>
              <v-list-item-title>{{ q.question }}</v-list-item-title>
            </v-list-item>
          </v-list>
        </v-card>
      </v-col>

      <!-- RIGHT: Selected FAQ Detail -->
      <!-- This is the answer area. Click a question, get an answer. Magic. -->
      <v-col cols="12" md="8" class="faq-detail">
        <v-card class="pa-6">
          <v-card-title class="text-h6 font-weight-bold d-flex align-center">
            <v-icon start color="secondary">mdi-book-open-page-variant</v-icon>
            <!-- Show the same question text. Again. For emphasis. -->
            {{ currentFaq.question }}
          </v-card-title>
          <v-divider class="my-4" />

          <v-card-text>
            <div class="text-body-1">
              {{ currentFaq.answer }}
              <!-- Sometimes, an icon is worth a thousand words. Or just one. -->
              <v-icon v-if="currentFaq.icon" class="mx-1" color="primary">{{
                currentFaq.icon
              }}</v-icon>
              {{ currentFaq.answerSuffix }}
            </div>

            <!-- If the answer wasn't clear enough, here are... steps. -->
            <div v-if="currentFaq.tutorial" class="mt-4">
              <h3 class="text-subtitle-1 font-weight-medium mb-2">Langkah-langkah:</h3>
              <v-list density="compact">
                <v-list-item
                  v-for="(step, idx) in currentFaq.tutorial"
                  :key="idx"
                  class="tutorial-step"
                >
                  <template #prepend>
                    <!-- Look, numbers! -->
                    <v-icon color="secondary">mdi-numeric-{{ idx + 1 }}-circle-outline</v-icon>
                  </template>
                  <v-list-item-title>{{ step }}</v-list-item-title>
                </v-list-item>
              </v-list>
            </div>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script setup>
// Yes, this is all hardcoded.
// No, we are not fetching this from a CMS.
// I am the CMS.
import { ref, computed } from 'vue'

// The sacred texts.
const faqs = [
  {
    question: 'Bagaimana cara memulai monitoring?',
    answer:
      'Buka halaman Monitor, pilih sumber kamera atau unggah file gambar, lalu tekan tombol Start Capture.',
    tutorial: [
      'Klik menu Monitor di navigasi.',
      'Pilih kamera atau unggah file.',
      'Tekan tombol Start Capture.',
      'Lihat hasil deteksi di panel log.',
    ],
  },
  {
    question: 'Bagaimana cara menyimpan log ke Riwayat?',
    answer:
      // This answer is probably a lie.
      // I think we made this automatic.
      // Did we? I... I don't remember.
      "Riwayat monitoring disimpan secara otomatis saat Anda menekan tombol 'Stop' di tab 'Live Monitoring'.",
  },
  {
    question: 'Apa fungsi halaman Model?',
    answer:
      'Halaman Model digunakan untuk mengatur atau mengganti model deteksi yang digunakan aplikasi.',
  },
  {
    question: 'Bagaimana cara mengganti tema gelap/terang?',
    answer: 'Gunakan tombol',
    icon: 'mdi-weather-night', // Or mdi-white-balance-sunny. Whatever.
    answerSuffix: 'di pojok kanan atas untuk beralih antara mode gelap dan terang.',
  },
]

// 'selected' just holds the index of the question they clicked.
// Default to 0, because arrays.
const selected = ref(0)
// 'currentFaq' is a "computed" property.
// It just means "get the FAQ at index [selected]".
// So revolutionary.
const currentFaq = computed(() => faqs[selected.value])
</script>

<style scoped>
.help-page {
  padding: 24px;
  /*
    ANOTHER ONE.
    My beautiful min-height calc.
    100vh - 64px - 80px = 144px.
    It's perfect. Don't touch it.
  */
  min-height: calc(100vh - 144px);
}

.faq-list {
  /* A faint line to separate the questions from the answers. */
  border-right: 1px solid rgba(0, 0, 0, 0.08);
}

.faq-item {
  margin-bottom: 4px;
  border-radius: 8px;
  transition: background-color 0.2s;
}
.faq-item:hover {
  /* Wow, a hover effect. */
  background-color: rgba(0, 0, 0, 0.04);
}
.faq-item.v-list-item--active {
  /* Wow, an active state. */
  background-color: rgba(var(--v-theme-primary), 0.12);
}

.faq-detail {
  padding-left: 16px;
}

.tutorial-step {
  border-bottom: 1px solid rgba(0, 0, 0, 0.06);
}

.v-dialog .v-card {
  border-radius: 12px;
}

/* Oh great, responsive styles. */
@media (max-width: 960px) {
  .faq-list {
    /* On mobile, the line goes on the bottom. */
    border-right: none;
    border-bottom: 1px solid rgba(0, 0, 0, 0.08);
    margin-bottom: 16px;
  }
  .faq-detail {
    padding-left: 0;
  }
}
</style>
