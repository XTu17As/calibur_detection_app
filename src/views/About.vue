<template>
  <!-- This whole page is basically just a glorified text file. -->
  <v-container class="about-page py-8">
    <v-row justify="center">
      <v-col cols="12" md="8">
        <v-card class="pa-6">
          <v-card-title class="text-h5 font-weight-bold">
            <v-icon start color="primary">mdi-information-outline</v-icon>
            Tentang Aplikasi
          </v-card-title>

          <v-divider class="my-4" />

          <v-card-text>
            <!-- A bunch of marketing fluff. -->
            <p class="mb-4">
              <strong>Deteksi Ketersediaan Rak</strong> adalah aplikasi berbasis web yang dirancang
              untuk membantu memantau ketersediaan rak secara otomatis. Aplikasi ini memanfaatkan
              kamera atau file gambar untuk melakukan deteksi, lalu menampilkan hasilnya dalam
              bentuk log dan riwayat.
            </p>

            <p class="mb-4">
              Tujuan utama aplikasi ini adalah memberikan <em>dashboard</em> yang sederhana namun
              informatif, sehingga pengguna dapat dengan cepat mengetahui kondisi rak, penggunaan
              sistem, serta status koneksi.
              <!-- "Sederhana". Right. -->
            </p>

            <p class="mb-4">
              Aplikasi ini dikembangkan oleh <strong>Richard</strong> sebagai proyek pembelajaran
              dan implementasi teknologi modern seperti <code>Vue 3</code>, <code>Vuetify</code>,
              <code>Pinia</code>, dan integrasi dengan API backend.
              <!-- Blame Richard. -->
            </p>

            <p>
              Jika Anda ingin mengetahui lebih lanjut atau memberikan masukan, silakan hubungi
              pengembang melalui halaman bantuan atau kontak yang tersedia.
            </p>
          </v-card-text>

          <v-divider class="my-4" />

          <v-card-actions>
            <!-- A button to the help page nobody will read. -->
            <v-btn color="primary" variant="tonal" to="/help">
              <v-icon start>mdi-help-circle-outline</v-icon>
              Bantuan
            </v-btn>
            <!-- This button opens the "blame someone else" dialog. -->
            <v-btn color="secondary" variant="tonal" @click="contactDialog = true">
              <v-icon start>mdi-card-account-phone-outline</v-icon>
              Kontak IT Support
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>
    </v-row>

    <!--
      This dialog is the "Kontak IT Support" popup.
      It's just placeholder text. Don't call these numbers.
      They probably connect to a cursed hotline.
    -->
    <v-dialog v-model="contactDialog" max-width="500px">
      <v-card>
        <v-card-title class="text-h6"> Kontak IT Support Perusahaan </v-card-title>
        <v-divider />
        <v-card-text>
          <v-list>
            <v-list-item
              prepend-icon="mdi-domain"
              title="IT Support - Gudang A"
              subtitle="it.gudanga@perusahaan.com"
            ></v-list-item>
            <v-list-item
              prepend-icon="mdi-domain"
              title="IT Support - Gudang B"
              subtitle="it.gudangb@perusahaan.com"
            ></v-list-item>
            <v-list-item
              prepend-icon="mdi-phone"
              title="Helpdesk Pusat"
              subtitle="(021) 555-1234"
            ></v-list-item>
          </v-list>
        </v-card-text>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn color="primary" variant="text" @click="contactDialog = false"> Tutup </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup>
// Import ref. For the dialog. Of course.
import { ref } from 'vue'

// This toggles the contact popup.
// contactDialog.value = true. contactDialog.value = false.
const contactDialog = ref(false)
</script>

<style scoped>
.about-page {
  min-height: calc(100vh - 144px);
}
</style>
