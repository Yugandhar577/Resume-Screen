import type { Metadata } from 'next'
import localFont from 'next/font/local'
import './globals.css'

const noto = localFont({ src: '../public/fonts/NotoSans-Regular.ttf', display: 'swap' })

export const metadata: Metadata = {
  title: 'CareerLens - Resume Analysis',
  description: 'Explainable skill coverage, reviewer feedback, and clean resume export',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={noto.className}>{children}</body>
    </html>
  )
}
