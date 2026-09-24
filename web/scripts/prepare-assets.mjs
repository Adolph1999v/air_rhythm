import { copyFileSync, mkdirSync, statSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const sourceDirectory = join(webRoot, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm')
const destinationDirectory = join(webRoot, 'public', 'wasm')
const files = [
  'vision_wasm_internal.js',
  'vision_wasm_internal.wasm',
  'vision_wasm_nosimd_internal.js',
  'vision_wasm_nosimd_internal.wasm',
]

mkdirSync(destinationDirectory, { recursive: true })

for (const name of files) {
  const source = join(sourceDirectory, name)
  const destination = join(destinationDirectory, name)
  const sourceInfo = statSync(source)
  const destinationInfo = statSync(destination, { throwIfNoEntry: false })
  if (!destinationInfo || sourceInfo.size !== destinationInfo.size || sourceInfo.mtimeMs > destinationInfo.mtimeMs) {
    copyFileSync(source, destination)
  }
}
