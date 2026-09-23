import { MELODY_STEPS, type InstrumentKey } from './music'

export const DEFAULT_TEMPO_SCALE = 1.75
export const DEFAULT_COUNTDOWN_SECONDS = 3
export const NODE_TRAVEL_SECONDS = 2
export const DEFAULT_LEAD_IN_SECONDS = NODE_TRAVEL_SECONDS
export const GOOD_WINDOW_SECONDS = 0.42

export type TimingGrade = 'PERFECT' | 'GREAT' | 'GOOD' | 'HIT'
export type MovementRating = 'GOOD' | 'GREAT' | 'PERFECT'
export type RoundPhase = 'COUNTDOWN' | 'PLAYING' | 'RESULTS'
export interface ChartEvent {
  index: number
  pitch: number
  targetOffset: number
  instrument: InstrumentKey
}

const GRADE_BONUS: Record<TimingGrade, number> = { PERFECT: 500, GREAT: 250, GOOD: 100, HIT: 0 }
const MOVEMENT_BONUS: Record<MovementRating, number> = { GOOD: 0, GREAT: 40, PERFECT: 80 }

export function buildMelodyChart(
  tempoScale = DEFAULT_TEMPO_SCALE,
  leadInSeconds = DEFAULT_LEAD_IN_SECONDS,
): ChartEvent[] {
  if (!Number.isFinite(tempoScale) || tempoScale <= 0) throw new RangeError('Tempo must be positive and finite.')
  if (!Number.isFinite(leadInSeconds) || leadInSeconds < 0) throw new RangeError('Lead-in must be non-negative and finite.')
  let targetOffset = leadInSeconds
  return MELODY_STEPS.map(([pitch, duration], index) => {
    const event: ChartEvent = { index, pitch, targetOffset, instrument: 'keys' }
    if (index < MELODY_STEPS.length - 1) targetOffset += duration * tempoScale
    return event
  })
}

export function gradeTiming(errorSeconds: number): TimingGrade | null {
  if (!Number.isFinite(errorSeconds)) throw new RangeError('Timing error must be finite.')
  const distance = Math.abs(errorSeconds)
  if (distance <= 0.10) return 'PERFECT'
  if (distance <= 0.22) return 'GREAT'
  if (distance <= GOOD_WINDOW_SECONDS) return 'GOOD'
  return null
}

export class RoundScore {
  score = 0
  combo = 0
  maxCombo = 0
  perfect = 0
  great = 0
  good = 0
  basicHits = 0
  misses = 0

  recordHit(grade: TimingGrade, movementRating: MovementRating = 'GOOD'): number {
    this.combo += 1
    this.maxCombo = Math.max(this.maxCombo, this.combo)
    if (grade === 'PERFECT') this.perfect += 1
    else if (grade === 'GREAT') this.great += 1
    else if (grade === 'GOOD') this.good += 1
    else this.basicHits += 1
    const comboBonus = Math.min((this.combo - 1) * 20, 200)
    const points = 1000 + GRADE_BONUS[grade] + comboBonus + MOVEMENT_BONUS[movementRating]
    this.score += points
    return points
  }

  recordMiss(): void {
    this.misses += 1
    this.combo = 0
  }

  get totalHits(): number { return this.perfect + this.great + this.good + this.basicHits }
  get totalJudged(): number { return this.totalHits + this.misses }
  get completionAccuracy(): number { return this.totalJudged ? this.totalHits / this.totalJudged * 100 : 0 }
  get timingAccuracy(): number {
    if (!this.totalJudged) return 0
    return (this.perfect + this.great * 0.75 + this.good * 0.5 + this.basicHits * 0.25) / this.totalJudged * 100
  }
  get rank(): string {
    if (this.completionAccuracy >= 90) return 'S'
    if (this.completionAccuracy >= 75) return 'A'
    if (this.completionAccuracy >= 60) return 'B'
    return 'C'
  }
}

export class RhythmRound {
  readonly chart: readonly ChartEvent[]
  readonly score = new RoundScore()
  private nextSpawnPosition = 0
  private readonly resolved = new Set<number>()
  private readonly orderPenalized = new Set<number>()
  private readonly chartPosition = new Map<number, number>()

  constructor(
    public startedAt: number,
    public readonly countdownSeconds = DEFAULT_COUNTDOWN_SECONDS,
    public readonly nodeTravelSeconds = NODE_TRAVEL_SECONDS,
    chart: readonly ChartEvent[] = buildMelodyChart(),
  ) {
    if (!Number.isFinite(startedAt) || !Number.isFinite(countdownSeconds) || countdownSeconds < 0 ||
        !Number.isFinite(nodeTravelSeconds) || nodeTravelSeconds <= 0) throw new RangeError('Invalid round clock.')
    this.chart = [...chart]
    let lastOffset = -1
    this.chart.forEach((event, position) => {
      if (this.chartPosition.has(event.index) || !Number.isFinite(event.targetOffset) ||
          event.targetOffset < 0 || event.targetOffset < lastOffset) throw new RangeError('Invalid chart order.')
      this.chartPosition.set(event.index, position)
      lastOffset = event.targetOffset
    })
  }

  get songStartTime(): number { return this.startedAt + this.countdownSeconds }
  get resolvedCount(): number { return this.resolved.size }
  get remainingCount(): number { return this.chart.length - this.resolvedCount }
  get nextUnresolvedIndex(): number | null {
    return this.chart.find(event => !this.resolved.has(event.index))?.index ?? null
  }
  phaseAt(now: number): RoundPhase {
    if (this.resolvedCount === this.chart.length) return 'RESULTS'
    return now < this.songStartTime ? 'COUNTDOWN' : 'PLAYING'
  }
  countdownRemaining(now: number): number { return Math.max(0, this.songStartTime - now) }
  progressAt(now: number): number {
    if (!Number.isFinite(now)) throw new RangeError('Time must be finite.')
    if (!this.chart.length || this.phaseAt(now) === 'RESULTS') return 1
    const duration = this.chart[this.chart.length - 1].targetOffset + GOOD_WINDOW_SECONDS
    return Math.max(0, Math.min(1, (now - this.songStartTime) / duration))
  }
  delayTimeline(seconds: number): void {
    if (!Number.isFinite(seconds) || seconds < 0) throw new RangeError('Pause must be non-negative and finite.')
    this.startedAt += seconds
  }
  targetTime(event: ChartEvent): number { return this.songStartTime + event.targetOffset }
  spawnTime(event: ChartEvent): number { return this.targetTime(event) - this.nodeTravelSeconds }
  dueEvents(now: number): ChartEvent[] {
    const due: ChartEvent[] = []
    while (this.nextSpawnPosition < this.chart.length && this.spawnTime(this.chart[this.nextSpawnPosition]) <= now) {
      due.push(this.chart[this.nextSpawnPosition++])
    }
    return due
  }
  timingBonusAvailable(index: number): boolean {
    return !this.resolved.has(index) && !this.orderPenalized.has(index) && index === this.nextUnresolvedIndex
  }
  judgeHit(event: ChartEvent, hitAt: number, movementRating: MovementRating): TimingGrade {
    if (this.resolved.has(event.index) || !Number.isFinite(hitAt)) throw new RangeError('Invalid or repeated hit.')
    const inOrder = this.timingBonusAvailable(event.index)
    if (!inOrder) {
      const position = this.chartPosition.get(event.index)
      if (position === undefined) throw new RangeError('Unknown chart event.')
      for (const earlier of this.chart.slice(0, position)) {
        if (!this.resolved.has(earlier.index)) this.orderPenalized.add(earlier.index)
      }
    }
    const grade = inOrder ? (gradeTiming(hitAt - this.targetTime(event)) ?? 'HIT') : 'HIT'
    this.resolved.add(event.index)
    this.score.recordHit(grade, movementRating)
    return grade
  }
  recordMiss(event: ChartEvent): boolean {
    if (this.resolved.has(event.index)) return false
    this.resolved.add(event.index)
    this.score.recordMiss()
    return true
  }
  isResolved(event: ChartEvent): boolean { return this.resolved.has(event.index) }
}
