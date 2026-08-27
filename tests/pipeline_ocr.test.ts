// @ts-nocheck
import { test, expect } from 'bun:test'
import { generateImageNarrationsOCR } from '../mini-services/pipeline-service/lib'

test('generateImageNarrationsOCR chunked batch processing and summary log', async () => {
  // Mock fetch to simulate PaddleOCR service endpoint
  const originalFetch = globalThis.fetch
  let requestCount = 0
  const requestedBodies: any[] = []

  globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
    if (url.toString().includes('/ocr/batch')) {
      requestCount++
      const body = JSON.parse(init?.body as string)
      requestedBodies.push(body)
      const numImages = body.images.length
      const mockResults = body.images.map((_: string, idx: number) => ({
        index: idx,
        text: `Solo Leveling Panel ${idx + 1} Text`,
        confidence: 0.95,
        regions: 1,
        status: 'SUCCESS',
      }))

      return new Response(
        JSON.stringify({
          results: mockResults,
          model: 'PP-OCRv5',
          processing_time_ms: 120,
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      )
    }
    return originalFetch(url, init)
  }) as typeof fetch

  try {
    const mockPanels = [
      '/tmp/solo_leveling_ch1_panel_1.png',
      '/tmp/solo_leveling_ch1_panel_2.png',
      '/tmp/solo_leveling_ch1_panel_3.png',
      '/tmp/solo_leveling_ch1_panel_4.png',
      '/tmp/solo_leveling_ch1_panel_5.png',
    ]

    const outcome = await generateImageNarrationsOCR(mockPanels)

    // 5 images with batchSize = 3 should produce 2 HTTP requests (3 + 2)
    expect(requestCount).toBe(2)
    expect(requestedBodies.length).toBe(2)
    expect(requestedBodies[0].images.length).toBe(3)
    expect(requestedBodies[1].images.length).toBe(2)
    expect(outcome.results.length).toBe(5)
    expect(outcome.results[0].text).toBe('Solo Leveling Panel 1 Text')
  } finally {
    globalThis.fetch = originalFetch
  }
})
