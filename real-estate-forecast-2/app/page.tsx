"use client"
import { useEffect, useState } from "react"
import dynamic from "next/dynamic"
import { Inter } from "next/font/google"
import { Building2, ChevronDown, BarChart3, Home } from "lucide-react"

const ForecastChart = dynamic(() => import("../components/ForecastChart"), { ssr: false })
const inter = Inter({ subsets: ["latin"] })

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL

export default function Page() {
  const [metros, setMetros] = useState<string[]>([])
  const [selectedMetro, setSelectedMetro] = useState("")
  const [forecast, setForecast] = useState<{ years: number[]; forecasted_prices: number[]; metro: string }>({
    years: [],
    forecasted_prices: [],
    metro: "",
  })
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    const fetchMetros = async () => {
      try {
        setIsLoading(true)
        const response = await fetch(`${API_BASE_URL}/api/metros`)
        const data = await response.json()
        if (Array.isArray(data)) {
          setMetros(data)
        } else if (data && data.metros) {
          setMetros(data.metros)
        } else {
          console.warn("Unexpected data format:", data)
          setMetros([])
        }
      } catch (error) {
        console.error("Error fetching metros:", error)
        setMetros([])
      } finally {
        setIsLoading(false)
      }
    }

    fetchMetros()
  }, [])

  useEffect(() => {
    const fetchForecast = async () => {
      if (selectedMetro) {
        try {
          setIsLoading(true)
          const response = await fetch(`${API_BASE_URL}/api/forecast/${encodeURIComponent(selectedMetro)}`)
          const data = await response.json()
          setForecast(data)
        } catch (error) {
          console.error("Error fetching forecast:", error)
          setForecast({ years: [], forecasted_prices: [], metro: "" })
        } finally {
          setIsLoading(false)
        }
      }
    }

    fetchForecast()
  }, [selectedMetro])

  return (
    <div className={`min-h-screen bg-[#0a0a0a] text-white ${inter.className}`}>
      {/* Header */}
      <header className="border-b border-zinc-800 bg-gradient-to-b from-[#0f0f0f] to-[#1a1a1a] shadow-lg">
        <div className="container mx-auto px-6 py-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Building2 className="h-8 w-8 text-cyan-400" />
            <h1 className="text-2xl font-extrabold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
              Real Estate Analytics
            </h1>
          </div>
          <nav className="hidden md:flex flex-col items-start gap-4">
            <a href="#" className="text-base text-zinc-300 hover:text-cyan-400 transition-colors duration-200">
              Dashboard
            </a>
            <a href="#" className="text-base text-zinc-300 hover:text-cyan-400 transition-colors duration-200">
              Markets
            </a>
            <a href="#" className="text-base text-zinc-300 hover:text-cyan-400 transition-colors duration-200">
              Insights
            </a>
            <a href="#" className="text-base text-zinc-300 hover:text-cyan-400 transition-colors duration-200">
              Settings
            </a>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-8">
            {/* Metro Selection Card */}
            <div className="card p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Home className="h-5 w-5 text-accent" />
                <span>Select Metro Area</span>
              </h3>
              <div className="relative">
                <select
                  className="w-full p-3 pr-10 rounded-lg bg-[#181c20] border border-cyan-400 text-white focus:border-accent focus:ring-2 focus:ring-accent focus:outline-none appearance-none transition-all duration-200"
                  value={selectedMetro}
                  onChange={(e) => setSelectedMetro(e.target.value)}
                  disabled={isLoading || metros.length === 0}
                >
                  <option value="">-- Choose a Metro --</option>
                  {metros.map((metro) => (
                    <option key={metro} value={metro}>{metro}</option>
                  ))}
                </select>
                <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 h-5 w-5 text-zinc-400 pointer-events-none" />
              </div>
              {isLoading && <div className="mt-3 text-sm text-zinc-400">Loading data...</div>}
            </div>
            {/* Stats Card */}
            {forecast.years && forecast.years.length > 0 && (
              <div className="card p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <BarChart3 className="h-5 w-5 text-accent" />
                  <span>Market Stats</span>
                </h3>
                <div className="space-y-4">
                  <div>
                    <p className="text-sm text-zinc-400">Selected Market</p>
                    <p className="text-lg font-medium">{forecast.metro}</p>
                  </div>
                  <div>
                    <p className="text-sm text-zinc-400">Current Price (est.)</p>
                    <p className="text-lg font-medium">${forecast.forecasted_prices[0]?.toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-sm text-zinc-400">Forecast Period</p>
                    <p className="text-lg font-medium">{forecast.years[0]} - {forecast.years[forecast.years.length - 1]}</p>
                  </div>
                </div>
              </div>
            )}
          </div>
          {/* Chart Card */}
          <div className="lg:col-span-2">
            <div className="card p-8 h-full flex flex-col">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-accent" />
                <span>Price Forecast</span>
              </h3>
              <div className="h-[400px] flex-1 flex items-center justify-center">
                {isLoading ? (
                  <div className="flex flex-col items-center justify-center text-zinc-400">
                    <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin mb-3"></div>
                    <p>Loading forecast data...</p>
                  </div>
                ) : forecast.years && forecast.years.length > 0 ? (
                  <div style={{ width: "100%", height: 400 }}>
                    <ForecastChart
                      years={forecast.years}
                      prices={forecast.forecasted_prices}
                      metro={forecast.metro}
                    />
                  </div>
                ) : (
                  <div className="text-zinc-400 text-center">
                    <p className="mb-2">No forecast data available</p>
                    <p className="text-sm">Select a metro area to view price forecasts</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-auto border-t border-zinc-800 bg-[#0f0f0f]">
        <div className="container mx-auto px-4 py-6">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <p className="text-sm text-zinc-500">© {new Date().getFullYear()} Real Estate Forecast Explorer</p>
            <div className="flex items-center gap-4 mt-4 md:mt-0">
              <a href="#" className="text-sm text-zinc-500 hover:text-cyan-400 transition-colors">
                Terms
              </a>
              <a href="#" className="text-sm text-zinc-500 hover:text-cyan-400 transition-colors">
                Privacy
              </a>
              <a href="#" className="text-sm text-zinc-500 hover:text-cyan-400 transition-colors">
                Contact
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}


