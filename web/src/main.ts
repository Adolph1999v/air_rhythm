import { attachCamera, cameraErrorMessage, requestCameraStream, stopCamera } from './camera'
import { drawInputPreview, StageRenderer } from './stage'
import { createHandTracker, trackedHandsFrom, type TrackedHand } from './tracking'
import type { HandLandmarker } from '@mediapipe/tasks-vision'
import './style.css'

function requiredElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector)
  if (!element) throw new Error(`The page is missing ${selector}.`)
  return element
}

const app = requiredElement<HTMLElement>('#app')
app.innerHTML = `
  <section class="experience" data-mode="idle">
    <canvas id="stage" class="stage" aria-label="A generated black starfield where tracked hands appear as virtual drumsticks"></canvas>
    <header class="topbar">
      <div class="brand">
        <span class="brand-icon" aria-hidden="true">✦</span>
        <span>AIR RHYTHM</span>
        <span class="brand-tag">WEB PREVIEW</span>
      </div>
      <div class="top-actions">
        <span id="camera-badge" class="badge">CAMERA OFF</span>
        <span id="model-badge" class="badge">MODEL IDLE</span>
        <span id="sound-badge" class="badge">SOUND OFF</span>
        <button id="stop-button" class="stop-button" type="button" hidden>Stop camera</button>
      </div>
    </header>
    <div id="welcome" class="welcome">
      <p class="eyebrow">LIVE RHYTHM STAGE · TRACKING PREVIEW</p>
      <h1>Play with your hands.</h1>
      <p class="intro-copy">
        Bring both hands into view. The same hand-landmark model used by the desktop app will move two virtual drumsticks in real time.
      </p>
      <button id="start-button" class="start-button" type="button">Enable camera &amp; tracking <span aria-hidden="true">↗</span></button>
      <p id="welcome-message" class="message" role="status" aria-live="polite">Camera is off. This app does not record or upload camera frames.</p>
    </div>
    <div id="live-guide" class="live-guide" hidden>
      <p class="eyebrow">TRACKING IS LIVE</p>
      <p>Move both hands and watch the sticks follow your index fingertips.</p>
      <span>Falling notes and hit sounds come in the next phase.</span>
    </div>
    <aside class="input-panel" aria-label="Live camera input">
      <div class="input-heading">
        <span><span class="live-dot" aria-hidden="true"></span> LIVE INPUT</span>
        <span id="hand-count">HANDS 0/2</span>
      </div>
      <div class="input-frame">
        <canvas id="input-preview" aria-label="Mirrored camera view with detected hand skeletons"></canvas>
        <div id="camera-placeholder" class="camera-placeholder">Camera preview appears here</div>
      </div>
    </aside>
    <footer class="footer">
      <p id="live-message" aria-live="polite">Camera is off. This app does not record or upload camera frames.</p>
      <p>PERSON-FREE MAIN STAGE <span aria-hidden="true">·</span> CAMERA ONLY IN LIVE INPUT</p>
    </footer>
    <video id="camera-source" autoplay muted playsinline hidden></video>
  </section>
`

type Mode = 'idle' | 'starting' | 'live' | 'error'

const experience = requiredElement<HTMLElement>('.experience')
const stageCanvas = requiredElement<HTMLCanvasElement>('#stage')
const inputPreview = requiredElement<HTMLCanvasElement>('#input-preview')
const video = requiredElement<HTMLVideoElement>('#camera-source')
const welcome = requiredElement<HTMLElement>('#welcome')
const liveGuide = requiredElement<HTMLElement>('#live-guide')
const startButton = requiredElement<HTMLButtonElement>('#start-button')
const stopButton = requiredElement<HTMLButtonElement>('#stop-button')
const cameraBadge = requiredElement<HTMLElement>('#camera-badge')
const modelBadge = requiredElement<HTMLElement>('#model-badge')
const soundBadge = requiredElement<HTMLElement>('#sound-badge')
const handCount = requiredElement<HTMLElement>('#hand-count')
const cameraPlaceholder = requiredElement<HTMLElement>('#camera-placeholder')
const welcomeMessage = requiredElement<HTMLElement>('#welcome-message')
const liveMessage = requiredElement<HTMLElement>('#live-message')

const stage = new StageRenderer(stageCanvas)
const mirroredFrame = document.createElement('canvas')
const mirroredContext = mirroredFrame.getContext('2d')
if (!mirroredContext) throw new Error('This browser cannot prepare camera frames.')

let stream: MediaStream | null = null
let tracker: HandLandmarker | null = null
let audioContext: AudioContext | null = null
let hands: TrackedHand[] = []
let lastVideoTime = -1
let sessionId = 0
let shownHandCount = -1
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')

function setMessage(message: string): void {
  welcomeMessage.textContent = message
  liveMessage.textContent = message
}

function setMode(mode: Mode): void {
  experience.dataset.mode = mode
  welcome.hidden = mode === 'live'
  liveGuide.hidden = mode !== 'live'
  startButton.disabled = mode === 'starting'
  stopButton.hidden = mode !== 'starting' && mode !== 'live'
}

function setHandCount(count: number): void {
  if (count === shownHandCount) return
  shownHandCount = count
  handCount.textContent = `HANDS ${count}/2`
  if (count === 2) setMessage('Two hands detected. Move your index fingertips to steer the sticks.')
  else if (count === 1) setMessage('One hand detected. Bring your other hand into the camera view.')
  else setMessage('Tracking is live. Bring both hands into the camera view.')
}

function endSession(message: string, mode: Mode = 'idle'): void {
  sessionId += 1
  tracker?.close()
  tracker = null
  stopCamera(video, stream)
  stream = null
  if (audioContext) void audioContext.close().catch(() => {})
  audioContext = null
  hands = []
  lastVideoTime = -1
  shownHandCount = -1
  inputPreview.getContext('2d')?.clearRect(0, 0, inputPreview.width, inputPreview.height)
  cameraPlaceholder.hidden = false
  cameraBadge.textContent = 'CAMERA OFF'
  modelBadge.textContent = 'MODEL IDLE'
  soundBadge.textContent = 'SOUND OFF'
  handCount.textContent = 'HANDS 0/2'
  setMode(mode)
  setMessage(message)
}

function beginAudio(token: number): void {
  if (typeof AudioContext === 'undefined') {
    soundBadge.textContent = 'SOUND UNAVAILABLE'
    return
  }
  try {
    const context = new AudioContext()
    audioContext = context
    soundBadge.textContent = 'SOUND STARTING'
    void context.resume().then(() => {
      if (token === sessionId) soundBadge.textContent = context.state === 'running' ? 'SOUND READY' : 'SOUND BLOCKED'
    }).catch(() => {
      if (token === sessionId) soundBadge.textContent = 'SOUND BLOCKED'
    })
  } catch {
    soundBadge.textContent = 'SOUND UNAVAILABLE'
  }
}

async function startSession(): Promise<void> {
  if (experience.dataset.mode === 'starting' || experience.dataset.mode === 'live') return
  const token = ++sessionId
  setMode('starting')
  setMessage('Waiting for camera permission…')
  beginAudio(token)

  try {
    const requestedStream = await requestCameraStream()
    if (token !== sessionId) {
      stopCamera(video, requestedStream)
      return
    }
    stream = requestedStream
    await attachCamera(video, requestedStream)
    if (token !== sessionId) return
    requestedStream.getVideoTracks()[0]?.addEventListener('ended', () => {
      if (token === sessionId) endSession('The camera disconnected. Reconnect it and try again.', 'error')
    }, { once: true })

    cameraBadge.textContent = 'CAMERA LIVE'
    cameraPlaceholder.hidden = true
    modelBadge.textContent = 'MODEL LOADING'
    setMessage('Camera is ready. Loading the local hand model…')
    const loadedTracker = await createHandTracker()
    if (token !== sessionId) {
      loadedTracker.close()
      return
    }
    tracker = loadedTracker
    modelBadge.textContent = 'MODEL READY'
    setMode('live')
    setHandCount(0)
  } catch (error) {
    if (token !== sessionId) return
    console.error('Air Rhythm web preview could not start:', error)
    endSession(cameraErrorMessage(error), 'error')
  }
}

function updateMirroredFrame(): void {
  const scale = Math.min(1, 960 / video.videoWidth, 720 / video.videoHeight)
  const width = Math.max(1, Math.round(video.videoWidth * scale))
  const height = Math.max(1, Math.round(video.videoHeight * scale))
  if (mirroredFrame.width !== width || mirroredFrame.height !== height) {
    mirroredFrame.width = width
    mirroredFrame.height = height
  }
  mirroredContext!.save()
  mirroredContext!.translate(width, 0)
  mirroredContext!.scale(-1, 1)
  mirroredContext!.drawImage(video, 0, 0, width, height)
  mirroredContext!.restore()
}

function animationFrame(timeMs: number): void {
  if (stream && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime
      try {
        updateMirroredFrame()
        if (tracker) {
          hands = trackedHandsFrom(tracker.detectForVideo(mirroredFrame, timeMs))
          setHandCount(hands.length)
        }
        drawInputPreview(inputPreview, mirroredFrame, hands)
      } catch (error) {
        console.error('Hand tracking stopped:', error)
        endSession('Hand tracking stopped unexpectedly. Please try again.', 'error')
      }
    }
  }
  stage.render(hands, reducedMotion.matches ? 0 : timeMs)
  requestAnimationFrame(animationFrame)
}

startButton.addEventListener('click', () => { void startSession() })
stopButton.addEventListener('click', () => endSession('Camera stopped. You can start it again.'))
window.addEventListener('pagehide', () => endSession('Camera stopped.'))
requestAnimationFrame(animationFrame)
