import { attachCamera, cameraErrorMessage, CameraFrameGate, requestCameraStream, stopCamera } from './camera'
import { BrowserAudio } from './audio'
import { RhythmGame } from './game'
import { FingertipMotionTracker, HandStabilizer, type FingertipMotion } from './hands'
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
  <section class="experience" data-mode="idle" data-screen="welcome">
    <canvas id="stage" class="stage" aria-label="Black starfield with falling notes and virtual drumsticks controlled by your hands"></canvas>
    <header class="topbar">
      <div class="brand"><span class="brand-icon" aria-hidden="true">✦</span><span>AIR RHYTHM</span><span class="brand-tag">WEB</span></div>
      <div class="top-actions">
        <span id="camera-badge" class="badge">CAMERA OFF</span>
        <span id="model-badge" class="badge">MODEL IDLE</span>
        <span id="sound-badge" class="badge">SOUND OFF</span>
        <button id="mute-button" class="quiet-button" type="button" hidden>Mute</button>
        <button id="help-button" class="quiet-button" type="button" hidden>Help</button>
        <button id="restart-button" class="quiet-button" type="button" hidden>Restart</button>
        <button id="menu-button" class="quiet-button" type="button" hidden>Menu</button>
        <button id="stop-button" class="quiet-button" type="button" hidden>Stop camera</button>
      </div>
    </header>
    <div id="welcome" class="welcome">
      <p class="eyebrow">LIVE RHYTHM STAGE · COMPUTER VISION</p>
      <h1>Play with your hands.</h1>
      <p class="intro-copy">Bring both hands into the camera view. The index fingertips steer virtual sticks; any fingertip can catch a falling note. Camera pixels stay in the live-input inset.</p>
      <button id="start-button" class="start-button" type="button">Enable camera &amp; tracking <span aria-hidden="true">↗</span></button>
      <p id="welcome-message" class="message" role="status" aria-live="polite">Camera is off. This app does not record or upload camera frames.</p>
    </div>
    <div id="menu" class="menu-panel" hidden>
      <p class="eyebrow">CAMERA READY · CHOOSE YOUR SET</p>
      <h1>Make the music move.</h1>
      <p class="intro-copy">Touch the falling circles with a fingertip. Move down or toward the camera for a stronger hit. Use both hands for the full melody.</p>
      <div class="menu-actions">
        <button id="challenge-button" class="start-button" type="button">Start song challenge <span aria-hidden="true">↗</span></button>
        <button id="free-button" class="secondary-button" type="button">Free play</button>
      </div>
      <p class="menu-caption">A short, hand-played version of Für Elise · 35 notes · no upload or recording</p>
    </div>
    <div id="hud" class="game-hud" hidden>
      <div class="hud-title"><span id="mode-name">SONG CHALLENGE</span><strong id="song-name">FÜR ELISE</strong><span id="note-progress">01 / 35</span></div>
      <div class="hud-values"><span>SCORE <strong id="score-value">0</strong></span><span>COMBO <strong id="combo-value">0</strong></span><span>HITS <strong id="hits-value">0</strong></span><span>MISSES <strong id="misses-value">0</strong></span></div>
      <div class="progress-track"><div id="progress-fill"></div></div>
      <p id="game-guide">Bring both hands into view. Touch circles with a fingertip; follow the note order for the best score.</p>
    </div>
    <div id="countdown" class="countdown" hidden aria-live="off"></div>
    <div id="results" class="results-panel" hidden>
      <p class="eyebrow">SONG COMPLETE</p>
      <h1 id="result-rank">RANK S</h1>
      <p id="result-score" class="result-score">0 POINTS</p>
      <p id="result-detail" class="result-detail"></p>
      <div class="menu-actions"><button id="retry-button" class="start-button" type="button">Play again <span aria-hidden="true">↗</span></button><button id="results-menu-button" class="secondary-button" type="button">Back to menu</button></div>
    </div>
    <div id="help" class="help-panel" hidden>
      <p class="eyebrow">PAUSED · HOW TO PLAY</p>
      <h2>Play with your index fingertips.</h2>
      <p>Bring both hands into the camera inset. The index fingertips steer the sticks. Any visible fingertip can touch a circle at any height until it leaves the bottom.</p>
      <p>In the song challenge, play circles in numbered order. A circle you touch out of order still sounds and scores a basic hit, but loses its timing bonus. Moving down or toward the camera adds a small movement bonus; a simple touch still works.</p>
      <p class="help-keys">1 Challenge · 2 Free play · R Restart · T Menu · H Help · M Mute</p>
      <button id="resume-button" class="start-button" type="button">Resume <span aria-hidden="true">↗</span></button>
    </div>
    <aside class="input-panel" aria-label="Live camera input">
      <div class="input-heading"><span><span class="live-dot" aria-hidden="true"></span> LIVE INPUT</span><span id="hand-count">HANDS 0/2</span></div>
      <div class="input-frame"><canvas id="input-preview" aria-label="Mirrored camera view with detected hand skeletons"></canvas><div id="camera-placeholder" class="camera-placeholder">Camera preview appears here</div></div>
    </aside>
    <footer class="footer"><p id="live-message" aria-live="polite">Camera is off. This app does not record or upload camera frames.</p><p>1 CHALLENGE · 2 FREE PLAY · H HELP · M MUTE · T MENU</p></footer>
    <video id="camera-source" autoplay muted playsinline hidden></video>
  </section>
`

type Mode = 'idle' | 'starting' | 'live' | 'error'
const experience = requiredElement<HTMLElement>('.experience')
const inputPreview = requiredElement<HTMLCanvasElement>('#input-preview')
const video = requiredElement<HTMLVideoElement>('#camera-source')
const welcome = requiredElement<HTMLElement>('#welcome')
const menu = requiredElement<HTMLElement>('#menu')
const hud = requiredElement<HTMLElement>('#hud')
const countdown = requiredElement<HTMLElement>('#countdown')
const results = requiredElement<HTMLElement>('#results')
const help = requiredElement<HTMLElement>('#help')
const startButton = requiredElement<HTMLButtonElement>('#start-button')
const stopButton = requiredElement<HTMLButtonElement>('#stop-button')
const muteButton = requiredElement<HTMLButtonElement>('#mute-button')
const helpButton = requiredElement<HTMLButtonElement>('#help-button')
const restartButton = requiredElement<HTMLButtonElement>('#restart-button')
const menuButton = requiredElement<HTMLButtonElement>('#menu-button')
const cameraBadge = requiredElement<HTMLElement>('#camera-badge')
const modelBadge = requiredElement<HTMLElement>('#model-badge')
const soundBadge = requiredElement<HTMLElement>('#sound-badge')
const handCount = requiredElement<HTMLElement>('#hand-count')
const cameraPlaceholder = requiredElement<HTMLElement>('#camera-placeholder')
const welcomeMessage = requiredElement<HTMLElement>('#welcome-message')
const liveMessage = requiredElement<HTMLElement>('#live-message')
const game = new RhythmGame()
const stage = new StageRenderer(requiredElement<HTMLCanvasElement>('#stage'))
const stabilizer = new HandStabilizer()
const motionTracker = new FingertipMotionTracker()
const cameraFrames = new CameraFrameGate()
const mirroredFrame = document.createElement('canvas')
const mirroredContext = mirroredFrame.getContext('2d')
if (!mirroredContext) throw new Error('This browser cannot prepare camera frames.')

let stream: MediaStream | null = null
let tracker: HandLandmarker | null = null
let audioContext: AudioContext | null = null
let audio: BrowserAudio | null = null
let activeHands: TrackedHand[] = []
let sessionId = 0
let shownHandCount = -1
let helpOpen = false
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')

function nowSeconds(): number { return performance.now() / 1000 }
function setText(selector: string, value: string): void {
  const element = requiredElement<HTMLElement>(selector)
  if (element.textContent !== value) element.textContent = value
}
function setMessage(message: string): void { welcomeMessage.textContent = message; liveMessage.textContent = message }

function updateScreen(now = nowSeconds()): void {
  const live = experience.dataset.mode === 'live'
  const screen = live ? game.screen : 'welcome'
  experience.dataset.screen = screen
  welcome.hidden = live
  menu.hidden = !live || screen !== 'menu' || helpOpen
  hud.hidden = !live || (screen !== 'challenge' && screen !== 'free')
  results.hidden = !live || screen !== 'results' || helpOpen
  help.hidden = !live || !helpOpen
  helpButton.hidden = !live || screen === 'menu'
  restartButton.hidden = !live || (screen !== 'challenge' && screen !== 'free')
  menuButton.hidden = !live || screen === 'menu'
  muteButton.hidden = !live
  if (screen === 'challenge' && game.round) {
    const remaining = game.round.countdownRemaining(now)
    countdown.hidden = remaining <= 0 || helpOpen
    countdown.textContent = String(Math.ceil(remaining))
  } else countdown.hidden = true
}

function setMode(mode: Mode): void {
  experience.dataset.mode = mode
  startButton.disabled = mode === 'starting'
  stopButton.hidden = mode !== 'starting' && mode !== 'live'
  updateScreen()
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
  tracker?.close(); tracker = null
  cameraFrames.stop()
  stopCamera(video, stream); stream = null
  audio?.stopAll(); audio = null
  if (audioContext) void audioContext.close().catch(() => {})
  audioContext = null
  activeHands = []
  stabilizer.reset(); motionTracker.reset(); game.toMenu()
  helpOpen = false; shownHandCount = -1
  inputPreview.getContext('2d')?.clearRect(0, 0, inputPreview.width, inputPreview.height)
  cameraPlaceholder.hidden = false
  cameraBadge.textContent = 'CAMERA OFF'
  modelBadge.textContent = 'MODEL IDLE'
  soundBadge.textContent = 'SOUND OFF'
  handCount.textContent = 'HANDS 0/2'
  setMode(mode); setMessage(message)
}

function beginAudio(token: number): void {
  if (typeof AudioContext === 'undefined') { soundBadge.textContent = 'SOUND UNAVAILABLE'; return }
  try {
    const context = new AudioContext()
    audioContext = context
    audio = new BrowserAudio(context)
    soundBadge.textContent = 'SOUND STARTING'
    void context.resume().then(() => {
      if (token === sessionId) soundBadge.textContent = context.state === 'running' ? 'SOUND READY' : 'SOUND BLOCKED'
    }).catch(() => { if (token === sessionId) soundBadge.textContent = 'SOUND BLOCKED' })
  } catch { soundBadge.textContent = 'SOUND UNAVAILABLE' }
}

async function startSession(): Promise<void> {
  if (experience.dataset.mode === 'starting' || experience.dataset.mode === 'live') return
  const token = ++sessionId
  setMode('starting'); setMessage('Waiting for camera permission…'); beginAudio(token)
  try {
    const requestedStream = await requestCameraStream()
    if (token !== sessionId) { stopCamera(video, requestedStream); return }
    stream = requestedStream
    await attachCamera(video, requestedStream)
    if (token !== sessionId) return
    cameraFrames.start(video, nowSeconds())
    requestedStream.getVideoTracks()[0]?.addEventListener('ended', () => {
      if (token === sessionId) endSession('The camera disconnected. Reconnect it and try again.', 'error')
    }, { once: true })
    cameraBadge.textContent = 'CAMERA LIVE'
    cameraPlaceholder.hidden = true
    modelBadge.textContent = 'MODEL LOADING'
    setMessage('Camera is ready. Loading the local hand model…')
    const loadedTracker = await createHandTracker()
    if (token !== sessionId) { loadedTracker.close(); return }
    tracker = loadedTracker
    modelBadge.textContent = 'MODEL READY'
    setMode('live'); setHandCount(0)
  } catch (error) {
    if (token !== sessionId) return
    console.error('Air Rhythm web session could not start:', error)
    endSession(cameraErrorMessage(error), 'error')
  }
}

function updateMirroredFrame(): void {
  // The desktop pipeline processes up to 1280×720; use the same landmark input size.
  const scale = Math.min(1, 1280 / video.videoWidth, 720 / video.videoHeight)
  const width = Math.max(1, Math.round(video.videoWidth * scale))
  const height = Math.max(1, Math.round(video.videoHeight * scale))
  if (mirroredFrame.width !== width || mirroredFrame.height !== height) {
    mirroredFrame.width = width; mirroredFrame.height = height
  }
  mirroredContext!.save()
  mirroredContext!.translate(width, 0)
  mirroredContext!.scale(-1, 1)
  mirroredContext!.drawImage(video, 0, 0, width, height)
  mirroredContext!.restore()
}

function startGame(mode: 'challenge' | 'free'): void {
  if (experience.dataset.mode !== 'live') return
  helpOpen = false
  audio?.stopAll()
  audio?.prepare(mode)
  if (audioContext?.state === 'suspended') void audioContext.resume().then(() => {
    soundBadge.textContent = audioContext?.state === 'running' ? 'SOUND READY' : 'SOUND BLOCKED'
  })
  motionTracker.reset()
  if (mode === 'challenge') game.startChallenge(nowSeconds())
  else game.startFree(nowSeconds())
  updateScreen()
}

function returnToMenu(): void {
  if (experience.dataset.mode !== 'live') return
  helpOpen = false
  audio?.stopAll()
  game.toMenu()
  updateScreen()
}

function toggleHelp(): void {
  if (experience.dataset.mode !== 'live' || game.screen === 'menu') return
  helpOpen = !helpOpen
  game.setPaused(helpOpen || document.hidden, nowSeconds())
  if (helpOpen) audio?.stopAll()
  updateScreen()
}

function toggleMute(): void {
  if (!audio) return
  audio.setMuted(!audio.muted)
  soundBadge.textContent = audio.muted ? 'SOUND MUTED' : 'SOUND READY'
  muteButton.textContent = audio.muted ? 'Unmute' : 'Mute'
}

function updateHud(now: number): void {
  if (game.screen === 'challenge' && game.round) {
    const score = game.round.score
    setText('#mode-name', 'SONG CHALLENGE')
    setText('#song-name', 'FÜR ELISE')
    setText('#score-value', score.score.toLocaleString())
    setText('#combo-value', String(score.combo))
    setText('#hits-value', String(score.totalHits))
    setText('#misses-value', String(score.misses))
    setText('#note-progress', `${String(Math.min(game.round.resolvedCount + 1, game.round.chart.length)).padStart(2, '0')} / ${game.round.chart.length}`)
    const fill = requiredElement<HTMLElement>('#progress-fill')
    fill.style.width = `${game.round.progressAt(now) * 100}%`
  } else if (game.screen === 'free') {
    setText('#mode-name', 'FREE PLAY')
    setText('#song-name', 'FOUR-INSTRUMENT FREE PLAY')
    setText('#score-value', String(game.freeHits * 100))
    setText('#combo-value', '0')
    setText('#hits-value', String(game.freeHits))
    setText('#misses-value', String(game.freeMisses))
    setText('#note-progress', 'NO TIME LIMIT')
    requiredElement<HTMLElement>('#progress-fill').style.width = '0%'
  } else if (game.screen === 'results' && game.round) {
    const score = game.round.score
    setText('#result-rank', `RANK ${score.rank}`)
    setText('#result-score', `${score.score.toLocaleString()} POINTS`)
    setText('#result-detail', `${score.totalHits} / ${game.round.chart.length} circles caught · ${score.completionAccuracy.toFixed(0)}% completion · ${score.timingAccuracy.toFixed(0)}% timing accuracy · best combo ${score.maxCombo}`)
  }
}

function animationFrame(timeMs: number): void {
  const now = timeMs / 1000
  const { width, height } = stage.size()
  let motions: FingertipMotion[] = []
  if (stream && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0 &&
      cameraFrames.consume(now)) {
    try {
      updateMirroredFrame()
      if (tracker) {
        activeHands = stabilizer.update(trackedHandsFrom(tracker.detectForVideo(mirroredFrame, timeMs)), now)
        setHandCount(activeHands.length)
        if (game.screen === 'challenge' || game.screen === 'free') {
          motions = motionTracker.update(activeHands, now, width, height)
        }
      }
      drawInputPreview(inputPreview, mirroredFrame, activeHands)
    } catch (error) {
      console.error('Hand tracking stopped:', error)
      endSession('Hand tracking stopped unexpectedly. Please try again.', 'error')
    }
  }
  const hits = game.update(now, width, height, motions)
  audio?.playHits(hits)
  updateHud(now)
  updateScreen(now)
  stage.render(stabilizer.visibleAt(now), timeMs, game, reducedMotion.matches)
  requestAnimationFrame(animationFrame)
}

startButton.addEventListener('click', () => { void startSession() })
stopButton.addEventListener('click', () => endSession('Camera stopped. You can start it again.'))
requiredElement<HTMLButtonElement>('#challenge-button').addEventListener('click', () => startGame('challenge'))
requiredElement<HTMLButtonElement>('#free-button').addEventListener('click', () => startGame('free'))
requiredElement<HTMLButtonElement>('#retry-button').addEventListener('click', () => startGame('challenge'))
requiredElement<HTMLButtonElement>('#results-menu-button').addEventListener('click', returnToMenu)
requiredElement<HTMLButtonElement>('#resume-button').addEventListener('click', toggleHelp)
helpButton.addEventListener('click', toggleHelp)
muteButton.addEventListener('click', toggleMute)
restartButton.addEventListener('click', () => startGame(game.screen === 'free' ? 'free' : 'challenge'))
menuButton.addEventListener('click', returnToMenu)
window.addEventListener('keydown', event => {
  if (experience.dataset.mode !== 'live' || event.repeat || event.metaKey || event.ctrlKey || event.altKey) return
  const key = event.key.toLowerCase()
  if (['1', '2', 'r', 't', 'escape', 'h', 'm', ' '].includes(key)) event.preventDefault()
  if (key === '1' || key === ' ') startGame('challenge')
  else if (key === '2') startGame('free')
  else if (key === 'r') startGame(game.screen === 'free' ? 'free' : 'challenge')
  else if (key === 't' || key === 'escape') returnToMenu()
  else if (key === 'h') toggleHelp()
  else if (key === 'm') toggleMute()
})
document.addEventListener('visibilitychange', () => {
  game.setPaused(helpOpen || document.hidden, nowSeconds())
  if (document.hidden) audio?.stopAll()
})
window.addEventListener('pagehide', () => endSession('Camera stopped.'))
requestAnimationFrame(animationFrame)
