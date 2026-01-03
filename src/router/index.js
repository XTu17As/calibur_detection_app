import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import Monitor from '../views/Monitor.vue'
import Model from '../views/Model.vue'
import History from '../views/History.vue'
import Help from '../views/Help.vue'
import About from '../views/About.vue'

const routes = [
  { path: '/', name: 'Home', component: Home },
  { path: '/monitor', name: 'Monitor', component: Monitor },
  { path: '/model', name: 'Model', component: Model },
  { path: '/history', name: 'History', component: History },
  { path: '/help', name: 'Help', component: Help },
  { path: '/about', name: 'About', component: About },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
