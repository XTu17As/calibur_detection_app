import CryptoJS from 'crypto-js'

const SECRET_KEY = 'rak-deteksi-2025'

export function encryptData(data) {
  const json = JSON.stringify(data)
  return CryptoJS.AES.encrypt(json, SECRET_KEY).toString()
}

export function decryptData(cipherText) {
  const bytes = CryptoJS.AES.decrypt(cipherText, SECRET_KEY)
  const decrypted = bytes.toString(CryptoJS.enc.Utf8)
  return JSON.parse(decrypted)
}
