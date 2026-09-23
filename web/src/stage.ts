import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import type { TrackedHand } from './tracking'
import type { FallingNode, RhythmGame } from './game'
import { nodeRadius } from './game'
import { INSTRUMENTS, noteName } from './music'

type Point = { x: number; y: number }
type Star = { x: number; y: number; radius: number; alpha: number }

const FINGERTIPS = [4, 8, 12, 16, 20] as const
const HAND_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
]

function colorFor(hand: TrackedHand, index: number): string {
  const label = hand.label.toLowerCase()
  if (label === 'left') return '#47ddff'
  if (label === 'right') return '#ff5ee0'
  return index % 2 === 0 ? '#47ddff' : '#ff5ee0'
}

function pointFor(landmark: NormalizedLandmark | undefined, width: number, height: number): Point | null {
  if (!landmark || !Number.isFinite(landmark.x) || !Number.isFinite(landmark.y)) return null
  return {
    x: Math.max(0, Math.min(1, landmark.x)) * width,
    y: Math.max(0, Math.min(1, landmark.y)) * height,
  }
}

function circle(ctx: CanvasRenderingContext2D, point: Point, radius: number): void {
  ctx.beginPath()
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2)
  ctx.fill()
}

function stroke(ctx: CanvasRenderingContext2D, start: Point, end: Point): void {
  ctx.beginPath()
  ctx.moveTo(start.x, start.y)
  ctx.lineTo(end.x, end.y)
  ctx.stroke()
}

export class StageRenderer {
  private readonly ctx: CanvasRenderingContext2D
  private width = 0
  private height = 0
  private stars: Star[] = []

  constructor(private readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('This browser cannot draw the performance stage.')
    this.ctx = ctx
  }

  size(): { width: number; height: number } {
    this.resizeIfNeeded()
    return { width: this.width, height: this.height }
  }

  render(hands: TrackedHand[], timeMs: number, game?: RhythmGame, reducedMotion = false): void {
    this.resizeIfNeeded()
    const { ctx, width, height } = this
    ctx.fillStyle = '#020305'
    ctx.fillRect(0, 0, width, height)

    for (const star of this.stars) {
      ctx.fillStyle = `rgba(220, 229, 245, ${star.alpha})`
      circle(ctx, { x: star.x * width, y: star.y * height }, star.radius)
    }

    const seconds = reducedMotion ? 0 : timeMs / 1000
    const center = {
      x: width * (0.50 + 0.10 * Math.sin(seconds * 0.55)),
      y: height * (0.48 + 0.05 * Math.cos(seconds * 0.46)),
    }
    for (const [fraction, alpha] of [[0.18, 0.085], [0.30, 0.055]]) {
      ctx.beginPath()
      ctx.arc(center.x, center.y, Math.max(25, Math.min(width, height) * fraction), 0, Math.PI * 2)
      ctx.strokeStyle = `rgba(220, 230, 245, ${alpha})`
      ctx.lineWidth = 1
      ctx.stroke()
    }

    for (const [x, y, offset] of [[0.13, 0.26, 0], [0.83, 0.18, 1.7], [0.21, 0.73, 3.1], [0.76, 0.68, 4.8]]) {
      const pulse = 0.5 + 0.5 * Math.sin(seconds * 1.2 + offset)
      ctx.fillStyle = `rgba(235, 242, 255, ${0.18 + pulse * 0.25})`
      circle(ctx, { x: width * x, y: height * y }, 1.4)
    }

    if (game) {
      for (const node of game.nodes) this.drawNode(node, Boolean(node.event) && game.round?.nextUnresolvedIndex === node.event?.index)
      for (const effect of game.effects) this.drawEffect(effect, timeMs / 1000)
    }

    hands.forEach((hand, index) => this.drawStick(hand, colorFor(hand, index)))
  }

  private drawNode(node: FallingNode, next: boolean): void {
    const { ctx, width, height } = this
    const x = node.xRatio * width
    const y = node.yRatio * height
    const radius = nodeRadius(width, height)
    if (y + radius < 0 || y - radius > height) return
    ctx.save()
    ctx.shadowColor = node.color
    ctx.shadowBlur = radius * (next ? 0.9 : 0.6)
    const fill = ctx.createRadialGradient(x - radius * 0.28, y - radius * 0.34, 1, x, y, radius)
    fill.addColorStop(0, '#ffffff')
    fill.addColorStop(0.16, node.color)
    fill.addColorStop(0.88, '#101925')
    fill.addColorStop(1, node.color)
    ctx.fillStyle = fill
    circle(ctx, { x, y }, radius)
    ctx.shadowBlur = 0
    ctx.strokeStyle = node.color
    ctx.lineWidth = Math.max(2, radius * 0.06)
    ctx.stroke()
    ctx.fillStyle = '#f8fbff'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.font = `800 ${Math.max(13, radius * 0.45)}px system-ui, sans-serif`
    const label = node.event ? noteName(node.pitch) : INSTRUMENTS.find(item => item.key === node.instrument)!.label
    const labelY = y - (node.event ? radius * 0.1 : 0)
    ctx.strokeStyle = '#06111b'
    ctx.lineWidth = Math.max(2, radius * 0.08)
    ctx.strokeText(label, x, labelY)
    ctx.fillText(label, x, labelY)
    if (node.event) {
      ctx.fillStyle = '#bed0df'
      ctx.font = `700 ${Math.max(10, radius * 0.22)}px system-ui, sans-serif`
      ctx.fillText(String(node.event.index + 1).padStart(2, '0'), x, y + radius * 0.38)
    }
    ctx.restore()
  }

  private drawEffect(effect: RhythmGame['effects'][number], now: number): void {
    const age = now - effect.at
    if (age < 0 || age > 0.35) return
    const { ctx, width, height } = this
    const x = effect.xRatio * width
    const y = effect.yRatio * height - age * height * 0.08
    ctx.save()
    ctx.globalAlpha = Math.max(0, 1 - age / 0.35)
    ctx.shadowColor = effect.color
    ctx.shadowBlur = 18
    ctx.fillStyle = '#f8fbff'
    ctx.font = `800 ${Math.max(14, Math.min(width, height) * 0.024)}px system-ui, sans-serif`
    ctx.textAlign = 'center'
    ctx.fillText(effect.label, x, y)
    if (effect.detail) {
      ctx.font = `700 ${Math.max(10, Math.min(width, height) * 0.013)}px system-ui, sans-serif`
      ctx.fillStyle = effect.color
      ctx.fillText(effect.detail, x, y + 18)
    }
    ctx.restore()
  }

  private resizeIfNeeded(): void {
    const width = Math.max(1, this.canvas.clientWidth)
    const height = Math.max(1, this.canvas.clientHeight)
    const ratio = Math.min(window.devicePixelRatio || 1, 2, Math.sqrt(4_000_000 / (width * height)))
    const pixelWidth = Math.max(1, Math.round(width * ratio))
    const pixelHeight = Math.max(1, Math.round(height * ratio))
    if (this.canvas.width === pixelWidth && this.canvas.height === pixelHeight &&
        this.width === width && this.height === height) return

    this.canvas.width = pixelWidth
    this.canvas.height = pixelHeight
    this.ctx.setTransform(pixelWidth / width, 0, 0, pixelHeight / height, 0, 0)
    this.width = width
    this.height = height
    this.rebuildStars()
  }

  private rebuildStars(): void {
    let seed = (Math.imul(this.width, 73856093) ^ Math.imul(this.height, 19349663)) >>> 0
    const random = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
      return seed / 0x100000000
    }
    const count = Math.max(24, Math.min(170, Math.floor(this.width * this.height / 6500)))
    this.stars = Array.from({ length: count }, () => ({
      x: random(),
      y: random(),
      radius: random() > 0.96 ? 1.7 : random() > 0.76 ? 1.0 : 0.55,
      alpha: 0.14 + random() * 0.28,
    }))
  }

  private drawStick(hand: TrackedHand, accent: string): void {
    const base = pointFor(hand.landmarks[5], this.width, this.height)
    const tip = pointFor(hand.landmarks[8], this.width, this.height)
    if (!base || !tip) return
    let dx = tip.x - base.x
    let dy = tip.y - base.y
    let distance = Math.hypot(dx, dy)
    if (distance < 4) {
      const middleBase = pointFor(hand.landmarks[9], this.width, this.height)
      const middleTip = pointFor(hand.landmarks[12], this.width, this.height)
      if (!middleBase || !middleTip) return
      dx = middleTip.x - middleBase.x
      dy = middleTip.y - middleBase.y
      distance = Math.hypot(dx, dy)
    }
    if (distance < 1) return

    const minimum = Math.min(this.width, this.height)
    const length = Math.max(minimum * 0.16, Math.min(minimum * 0.38, distance * 1.85))
    const handle = { x: tip.x - dx / distance * length, y: tip.y - dy / distance * length }
    const radius = Math.max(4, minimum * 0.0105)
    const { ctx } = this
    ctx.save()
    ctx.lineCap = 'round'
    ctx.shadowColor = accent
    ctx.shadowBlur = radius * 3.8
    ctx.strokeStyle = accent
    ctx.globalAlpha = 0.3
    ctx.lineWidth = radius * 4
    stroke(ctx, handle, tip)
    ctx.globalAlpha = 1
    ctx.shadowBlur = 0
    ctx.strokeStyle = '#080d15'
    ctx.lineWidth = radius * 2 + 5
    stroke(ctx, handle, tip)
    ctx.strokeStyle = '#e09a58'
    ctx.lineWidth = radius * 2
    stroke(ctx, handle, tip)
    ctx.strokeStyle = '#f9e2c2'
    ctx.lineWidth = Math.max(2, radius * 0.5)
    stroke(ctx, handle, tip)
    ctx.fillStyle = accent
    circle(ctx, tip, radius + 1)
    ctx.fillStyle = '#f8fbff'
    circle(ctx, tip, Math.max(2, radius * 0.5))
    ctx.restore()

    const fingerRadius = Math.max(2, minimum * 0.0042)
    for (const index of FINGERTIPS) {
      const point = pointFor(hand.landmarks[index], this.width, this.height)
      if (!point) continue
      ctx.save()
      ctx.shadowColor = accent
      ctx.shadowBlur = fingerRadius * 4
      ctx.fillStyle = accent
      circle(ctx, point, fingerRadius)
      ctx.restore()
    }
  }
}

export function drawInputPreview(
  canvas: HTMLCanvasElement,
  mirroredFrame: HTMLCanvasElement,
  hands: TrackedHand[],
): void {
  if (canvas.width !== mirroredFrame.width || canvas.height !== mirroredFrame.height) {
    canvas.width = mirroredFrame.width
    canvas.height = mirroredFrame.height
    canvas.style.aspectRatio = `${canvas.width} / ${canvas.height}`
  }
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.drawImage(mirroredFrame, 0, 0)

  hands.forEach((hand, index) => {
    const accent = colorFor(hand, index)
    ctx.lineWidth = Math.max(2, canvas.width * 0.003)
    ctx.lineCap = 'round'
    ctx.strokeStyle = accent
    for (const [startIndex, endIndex] of HAND_CONNECTIONS) {
      const start = pointFor(hand.landmarks[startIndex], canvas.width, canvas.height)
      const end = pointFor(hand.landmarks[endIndex], canvas.width, canvas.height)
      if (start && end) stroke(ctx, start, end)
    }
    ctx.fillStyle = '#f8fbff'
    for (const landmark of hand.landmarks) {
      const point = pointFor(landmark, canvas.width, canvas.height)
      if (point) circle(ctx, point, Math.max(2, canvas.width * 0.0035))
    }
  })
}
