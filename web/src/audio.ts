import { INSTRUMENTS, MELODY_STEPS, type InstrumentKey } from './music'
import type { NodeHit } from './game'

type Partial = readonly [multiplier: number, amplitude: number, decay: number]
interface Timbre { duration: number; partials: readonly Partial[] }
const TIMBRES: Record<InstrumentKey, Timbre> = {
  keys: { duration: 1.5, partials: [[1, 1, 0.55], [2, 0.30, 0.28], [3, 0.10, 0.16]] },
  bell: { duration: 1.8, partials: [[1, 1, 0.70], [2, 0.35, 0.45], [3, 0.14, 0.23], [4.2, 0.06, 0.15]] },
  pluck: { duration: 1.0, partials: [[1, 1, 0.29], [2, 0.40, 0.15], [3, 0.17, 0.09], [4, 0.08, 0.06]] },
  marimba: { duration: 0.9, partials: [[1, 1, 0.25], [4, 0.28, 0.08], [10, 0.05, 0.03]] },
}

export class BrowserAudio {
  private readonly buffers = new Map<string, AudioBuffer>()
  private readonly sources = new Set<AudioBufferSourceNode>()
  muted = false

  constructor(private readonly context: AudioContext) {}

  prepare(mode: 'challenge' | 'free'): void {
    const notes: ReadonlyArray<readonly [InstrumentKey, number]> = mode === 'challenge'
      ? [...new Set(MELODY_STEPS.map(([pitch]) => pitch))].map(pitch => ['keys', pitch] as const)
      : INSTRUMENTS.map(instrument => [instrument.key, instrument.freestyleNote] as const)
    for (const [instrument, pitch] of notes) this.bufferFor(instrument, pitch)
  }

  playHits(hits: NodeHit[]): void {
    if (this.muted || this.context.state !== 'running') return
    const unique = new Map<string, NodeHit>()
    for (const hit of hits) {
      const key = `${hit.instrument}:${hit.pitch}`
      const earlier = unique.get(key)
      if (!earlier || this.velocity(hit) > this.velocity(earlier)) unique.set(key, hit)
    }
    for (const hit of unique.values()) this.play(hit.instrument, hit.pitch, this.velocity(hit))
  }

  setMuted(muted: boolean): void { this.muted = muted; this.stopAll() }

  stopAll(): void {
    for (const source of this.sources) {
      try { source.stop() } catch { /* A source may already have ended. */ }
      source.disconnect()
    }
    this.sources.clear()
  }

  private velocity(hit: NodeHit): number {
    return hit.movement === 'PERFECT' ? 1 : hit.movement === 'GREAT' ? 0.9 : 0.75
  }

  private play(instrument: InstrumentKey, pitch: number, velocity: number): void {
    const buffer = this.bufferFor(instrument, pitch)
    const source = this.context.createBufferSource()
    const gain = this.context.createGain()
    source.buffer = buffer
    gain.gain.value = 0.45 * velocity
    source.connect(gain).connect(this.context.destination)
    source.addEventListener('ended', () => { this.sources.delete(source); source.disconnect(); gain.disconnect() }, { once: true })
    this.sources.add(source)
    source.start()
  }

  private bufferFor(instrument: InstrumentKey, pitch: number): AudioBuffer {
    const key = `${instrument}:${pitch}`
    let buffer = this.buffers.get(key)
    if (!buffer) {
      buffer = this.synthesize(instrument, pitch)
      this.buffers.set(key, buffer)
    }
    return buffer
  }

  private synthesize(instrument: InstrumentKey, pitch: number): AudioBuffer {
    const { duration, partials } = TIMBRES[instrument]
    const sampleRate = this.context.sampleRate
    const count = Math.max(2, Math.round(duration * sampleRate))
    const buffer = this.context.createBuffer(1, count, sampleRate)
    const samples = buffer.getChannelData(0)
    const frequency = 440 * 2 ** ((pitch - 69) / 12)
    let peak = 0
    for (let index = 0; index < count; index += 1) {
      const time = index / sampleRate
      let value = 0
      for (const [multiplier, amplitude, decay] of partials) {
        if (frequency * multiplier < sampleRate * 0.48) {
          value += amplitude * Math.sin(2 * Math.PI * frequency * multiplier * time) * Math.exp(-time / decay)
        }
      }
      const attack = Math.min(1, index / Math.max(2, Math.round(sampleRate * 0.004)))
      const release = Math.min(1, (count - index - 1) / Math.max(2, Math.round(sampleRate * 0.03)))
      samples[index] = value * Math.max(0, attack) * Math.max(0, release)
      peak = Math.max(peak, Math.abs(samples[index]))
    }
    if (peak > 0) for (let index = 0; index < count; index += 1) samples[index] /= peak
    return buffer
  }
}
