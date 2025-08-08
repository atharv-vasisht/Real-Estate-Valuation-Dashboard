import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  output: 'export', // Enables static HTML export
  images: {
    unoptimized: true // Required for static export if you're using <Image>
  }
}

export default nextConfig
