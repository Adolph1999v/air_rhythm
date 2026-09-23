import './style.css'

const app = document.querySelector<HTMLElement>('#app')

if (!app) {
  throw new Error('The Air Rhythm app container is missing.')
}

app.innerHTML = `
  <section class="foundation" aria-labelledby="title">
    <p class="eyebrow">Web version · foundation</p>
    <h1 id="title">Air Rhythm</h1>
    <p class="description">
      The browser version is being built alongside the finished desktop app.
      Camera tracking and gameplay are coming in the next phases.
    </p>
  </section>
`
