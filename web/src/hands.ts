import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import type { TrackedHand } from './tracking'

export const FINGERTIPS = [4, 8, 12, 16, 20] as const
const TWO_PI = 2 * Math.PI
const MIN_CUTOFF = 2.2
const SPEED_COEFFICIENT = 1.2
const RESET_SECONDS = 0.25
const RETAIN_SECONDS = 1
const DROPOUT_GRACE_SECONDS = 0.14

function alpha(cutoff: number, dt: number): number {
  return 1 / (1 + 1 / (TWO_PI * cutoff * dt))
}

class LandmarkFilter {
  private lastTime: number | null = null
  private raw: number[][] = []
  private filtered: number[][] = []
  private derivative: number[][] = []

  update(landmarks: NormalizedLandmark[], now: number): NormalizedLandmark[] {
    const coordinates = landmarks.map(point => [point.x, point.y, point.z ?? 0])
    if (this.lastTime === null || now - this.lastTime <= 0 || now - this.lastTime > RESET_SECONDS ||
        coordinates.length !== this.raw.length) {
      this.lastTime = now
      this.raw = coordinates
      this.filtered = coordinates
      this.derivative = coordinates.map(() => [0, 0, 0])
      return landmarks.map(point => ({ ...point }))
    }
    const dt = Math.min(0.1, now - this.lastTime)
    const derivativeWeight = alpha(1, dt)
    const next = coordinates.map((point, index) => {
      const derivative = point.map((value, axis) => derivativeWeight * ((value - this.raw[index][axis]) / dt) +
        (1 - derivativeWeight) * this.derivative[index][axis])
      this.derivative[index] = derivative
      const planarSpeed = Math.hypot(derivative[0], derivative[1])
      return point.map((value, axis) => {
        const speed = axis === 2 ? Math.abs(derivative[2]) : planarSpeed
        const weight = alpha(MIN_CUTOFF + SPEED_COEFFICIENT * speed, dt)
        return weight * value + (1 - weight) * this.filtered[index][axis]
      })
    })
    this.lastTime = now
    this.raw = coordinates
    this.filtered = next
    return landmarks.map((point, index) => ({ ...point, x: next[index][0], y: next[index][1], z: next[index][2] }))
  }
}

interface HandTrack {
  identity: string
  filter: LandmarkFilter
  hand: TrackedHand
  wrist: { x: number; y: number }
  lastSeen: number
}

export class HandStabilizer {
  private readonly tracks = new Map<string, HandTrack>()
  private nextUnknown = 1

  update(rawHands: TrackedHand[], now: number): TrackedHand[] {
    for (const [identity, track] of this.tracks) {
      if (now - track.lastSeen > RETAIN_SECONDS) this.tracks.delete(identity)
    }
    const assigned = new Set<string>()
    const ordered = [...rawHands].sort((a, b) => a.label.localeCompare(b.label))
    const active: TrackedHand[] = []
    for (const raw of ordered) {
      const label = /^(left|right)$/i.test(raw.label) ? raw.label[0].toUpperCase() + raw.label.slice(1).toLowerCase() : null
      const available = [...this.tracks.values()].filter(track => !assigned.has(track.identity))
      const nearest = available.map(track => ({ track, distance: Math.hypot(
        raw.landmarks[0].x - track.wrist.x, raw.landmarks[0].y - track.wrist.y,
      ) })).sort((a, b) => a.distance - b.distance)[0]
      const labelled = label ? this.tracks.get(label) : undefined
      let identity: string
      if (label && labelled && !assigned.has(label) && (raw.confidence ?? 0) >= 0.65) identity = label
      else if (label && labelled && !assigned.has(label) && (!nearest || nearest.distance > 0.28 ||
        Math.hypot(raw.landmarks[0].x - labelled.wrist.x, raw.landmarks[0].y - labelled.wrist.y) <= 0.45)) identity = label
      else if (nearest && nearest.distance <= 0.28) identity = nearest.track.identity
      else if (label && !this.tracks.has(label)) identity = label
      else identity = `Hand-${this.nextUnknown++}`

      let track = this.tracks.get(identity)
      if (!track) {
        track = { identity, filter: new LandmarkFilter(), hand: raw, wrist: { x: 0, y: 0 }, lastSeen: now }
        this.tracks.set(identity, track)
      }
      const hand = { ...raw, identity, landmarks: track.filter.update(raw.landmarks, now) }
      track.hand = hand
      track.wrist = { x: raw.landmarks[0].x, y: raw.landmarks[0].y }
      track.lastSeen = now
      assigned.add(identity)
      active.push(hand)
    }
    return active.sort((a, b) => (a.identity ?? '').localeCompare(b.identity ?? ''))
  }

  visibleAt(now: number): TrackedHand[] {
    return [...this.tracks.values()].filter(track => now - track.lastSeen <= DROPOUT_GRACE_SECONDS)
      .sort((a, b) => a.identity.localeCompare(b.identity)).map(track => track.hand)
  }

  reset(): void { this.tracks.clear(); this.nextUnknown = 1 }
}

export interface FingertipMotion {
  previous: { x: number; y: number }
  current: { x: number; y: number }
  speed: number
  rating: 'GOOD' | 'GREAT' | 'PERFECT'
}

interface History {
  x: number; y: number; z: number
  pixel: { x: number; y: number }
  time: number; speed: number; downward: number; forward: number
}

export class FingertipMotionTracker {
  private history = new Map<string, History>()
  private dimensions = ''

  update(hands: TrackedHand[], now: number, width: number, height: number): FingertipMotion[] {
    const dimensions = `${width}x${height}`
    if (dimensions !== this.dimensions) this.history.clear()
    this.dimensions = dimensions
    const next = new Map<string, History>()
    const motions: FingertipMotion[] = []
    hands.forEach((hand, handIndex) => {
      for (const fingertip of FINGERTIPS) {
        const landmark = hand.landmarks[fingertip]
        const key = `${hand.identity ?? hand.label ?? handIndex}:${fingertip}`
        const previous = this.history.get(key)
        const pixel = { x: Math.max(0, Math.min(width, landmark.x * width)), y: Math.max(0, Math.min(height, landmark.y * height)) }
        let speed = 0; let downward = 0; let forward = 0
        let previousPixel = pixel
        if (previous && now - previous.time > 0 && now - previous.time <= 0.2) {
          const dt = now - previous.time
          const rawSpeed = Math.hypot(landmark.x - previous.x, landmark.y - previous.y) / dt
          speed = 0.6 * rawSpeed + 0.4 * previous.speed
          downward = 0.6 * ((landmark.y - previous.y) / dt) + 0.4 * previous.downward
          forward = 0.6 * ((previous.z - landmark.z) / dt) + 0.4 * previous.forward
          previousPixel = previous.pixel
        }
        const rating = downward >= 0.55 || forward >= 0.4 || speed >= 0.65 ? 'PERFECT' :
          speed >= 0.28 ? 'GREAT' : 'GOOD'
        motions.push({ previous: previousPixel, current: pixel, speed, rating })
        next.set(key, { x: landmark.x, y: landmark.y, z: landmark.z, pixel, time: now, speed, downward, forward })
      }
    })
    this.history = next
    return motions
  }

  reset(): void { this.history.clear(); this.dimensions = '' }
}

export function motionTouchesCircle(
  motion: FingertipMotion, center: { x: number; y: number }, radius: number,
): boolean {
  const dx = motion.current.x - motion.previous.x
  const dy = motion.current.y - motion.previous.y
  const lengthSquared = dx * dx + dy * dy
  const projection = lengthSquared ? Math.max(0, Math.min(1,
    ((center.x - motion.previous.x) * dx + (center.y - motion.previous.y) * dy) / lengthSquared,
  )) : 0
  const x = motion.previous.x + projection * dx
  const y = motion.previous.y + projection * dy
  return Math.hypot(center.x - x, center.y - y) <= radius
}
