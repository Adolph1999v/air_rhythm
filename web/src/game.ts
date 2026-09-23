import { motionTouchesCircle, type FingertipMotion } from './hands'
import { INSTRUMENTS, MELODY_TITLE, type InstrumentKey } from './music'
import { RhythmRound, type ChartEvent, type MovementRating, type TimingGrade } from './rhythm'

export type GameScreen = 'menu' | 'challenge' | 'free' | 'results'
export interface FallingNode {
  id: number
  xRatio: number
  yRatio: number
  instrument: InstrumentKey
  color: string
  pitch: number
  spawnedAt: number
  event?: ChartEvent
}
export interface HitEffect {
  xRatio: number
  yRatio: number
  color: string
  label: string
  detail: string
  at: number
}
export interface NodeHit {
  instrument: InstrumentKey
  pitch: number
  movement: MovementRating
  grade: TimingGrade | null
}

const SPAWN_INTERVAL_SECONDS = 0.55
const FREE_FALL_PER_SECOND = 0.28
const CHALLENGE_TARGET_RATIO = 0.30
const EFFECT_SECONDS = 0.35

export function nodeRadius(width: number, height: number): number {
  return Math.max(34, Math.min(width, height) * 0.0585)
}

export function challengeNodeY(node: FallingNode, now: number, width: number, height: number, round: RhythmRound): number {
  if (!node.event) throw new Error('Challenge node has no chart event.')
  const radiusRatio = nodeRadius(width, height) / height
  const speedRatio = Math.max(1 / height, (CHALLENGE_TARGET_RATIO - radiusRatio) / round.nodeTravelSeconds)
  return radiusRatio + speedRatio * Math.max(0, now - round.spawnTime(node.event))
}

export class RhythmGame {
  screen: GameScreen = 'menu'
  readonly songTitle = MELODY_TITLE
  readonly nodes: FallingNode[] = []
  readonly effects: HitEffect[] = []
  round: RhythmRound | null = null
  freeHits = 0
  freeMisses = 0
  private lastSpawnTime = 0
  private nextId = 1
  private pausedAt: number | null = null

  constructor(private readonly random: () => number = Math.random) {}

  startChallenge(now: number): void {
    this.clear()
    this.screen = 'challenge'
    this.round = new RhythmRound(now)
  }

  startFree(now: number): void {
    this.clear()
    this.screen = 'free'
    this.lastSpawnTime = now - SPAWN_INTERVAL_SECONDS
  }

  toMenu(): void {
    this.clear()
    this.screen = 'menu'
  }

  setPaused(paused: boolean, now: number): void {
    if (paused && this.pausedAt === null) this.pausedAt = now
    else if (!paused && this.pausedAt !== null) {
      const duration = now - this.pausedAt
      this.round?.delayTimeline(duration)
      this.lastSpawnTime += duration
      for (const node of this.nodes) node.spawnedAt += duration
      this.pausedAt = null
    }
  }

  update(now: number, width: number, height: number, motions: FingertipMotion[] = []): NodeHit[] {
    if (this.pausedAt !== null || (this.screen !== 'challenge' && this.screen !== 'free')) return []
    const radius = nodeRadius(width, height)
    const radiusRatio = radius / height
    const hits: NodeHit[] = []
    this.effects.splice(0, this.effects.length, ...this.effects.filter(effect => now - effect.at < EFFECT_SECONDS))

    if (this.screen === 'challenge' && this.round) {
      for (const event of this.round.dueEvents(now)) this.spawn(event, now, width, height)
    } else if (this.screen === 'free' && now - this.lastSpawnTime >= SPAWN_INTERVAL_SECONDS && this.nodes.length < 6) {
      this.spawn(undefined, now, width, height)
      this.lastSpawnTime = now
    }

    const remaining: FallingNode[] = []
    for (const node of this.nodes) {
      node.yRatio = node.event && this.round ? challengeNodeY(node, now, width, height, this.round) :
        radiusRatio + FREE_FALL_PER_SECOND * Math.max(0, now - node.spawnedAt)
      const center = { x: node.xRatio * width, y: node.yRatio * height }
      const touching = motions.filter(motion => motionTouchesCircle(motion, center, radius + 10))
      if (touching.length) {
        const movement = touching.some(motion => motion.rating === 'PERFECT') ? 'PERFECT' :
          touching.some(motion => motion.rating === 'GREAT') ? 'GREAT' : 'GOOD'
        let grade: TimingGrade | null = null
        let label: string = movement === 'PERFECT' ? 'POWER' : movement === 'GREAT' ? 'STRONG' : 'TOUCH'
        let detail = ''
        if (node.event && this.round) {
          const inOrder = this.round.timingBonusAvailable(node.event.index)
          grade = this.round.judgeHit(node.event, now, movement)
          label = grade
          detail = inOrder ? `NOTE ${node.event.index + 1}` : 'OUT OF ORDER'
        } else this.freeHits += 1
        this.effects.push({ xRatio: node.xRatio, yRatio: node.yRatio, color: node.color, label, detail, at: now })
        hits.push({ instrument: node.instrument, pitch: node.pitch, movement, grade })
      } else if (node.yRatio - radiusRatio > 1) {
        if (node.event && this.round) this.round.recordMiss(node.event)
        else this.freeMisses += 1
      } else remaining.push(node)
    }
    this.nodes.splice(0, this.nodes.length, ...remaining)
    if (this.round?.phaseAt(now) === 'RESULTS' && this.nodes.length === 0) this.screen = 'results'
    return hits
  }

  private spawn(event: ChartEvent | undefined, now: number, width: number, height: number): void {
    const radius = nodeRadius(width, height)
    const nearTop = this.nodes.filter(node => node.yRatio * height < radius * (event ? 5 : 4))
    const latestSongNode = [...this.nodes].reverse().find(node => node.event)
    const minX = event && latestSongNode ? Math.max(radius, latestSongNode.xRatio * width - width * 0.20) : radius
    const maxX = event && latestSongNode ? Math.min(width - radius, latestSongNode.xRatio * width + width * 0.20) : width - radius
    let x = minX + this.random() * Math.max(0, maxX - minX)
    for (let retry = 0; retry < 12; retry += 1) {
      if (nearTop.every(node => Math.abs(x - node.xRatio * width) >= radius * (event ? 2.5 : 3))) break
      x = minX + this.random() * Math.max(0, maxX - minX)
    }
    const instrument = event ? INSTRUMENTS[0] : INSTRUMENTS[Math.floor(this.random() * INSTRUMENTS.length)]
    this.nodes.push({
      id: this.nextId++, xRatio: x / width, yRatio: radius / height,
      instrument: instrument.key, color: instrument.color,
      pitch: event?.pitch ?? instrument.freestyleNote,
      spawnedAt: event && this.round ? this.round.spawnTime(event) : now, event,
    })
  }

  private clear(): void {
    this.nodes.length = 0
    this.effects.length = 0
    this.round = null
    this.freeHits = 0
    this.freeMisses = 0
    this.pausedAt = null
  }
}
