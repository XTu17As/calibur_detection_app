import 'vuetify/styles'
import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'
import { aliases, mdi } from 'vuetify/iconsets/mdi'

const lightTheme = {
  dark: false,
  colors: {
    primary: '#8BC34A', // Light Green
    secondary: '#FFC107', // Amber
    success: '#4CAF50',
    info: '#2196F3',
    warning: '#FB8C00',
    error: '#B00020',
    background: '#f5f5f5', // Light grey background
    surface: '#FFFFFF', // White card backgrounds
  },
}

const darkTheme = {
  dark: true,
  colors: {
    primary: '#9CCC65', // Lighter Green for dark mode
    secondary: '#FFD54F', // Lighter Amber
    success: '#81C784',
    info: '#64B5F6',
    warning: '#FFB74D',
    error: '#E57373',
    background: '#121212', // Standard dark background
    surface: '#1E1E1E', // Slightly lighter dark card backgrounds
  },
}

export default createVuetify({
  components,
  directives,
  icons: {
    defaultSet: 'mdi',
    aliases,
    sets: {
      mdi,
    },
  },

  theme: {
    defaultTheme: 'light',
    themes: {
      light: lightTheme,
      dark: darkTheme,
    },
  },
})
