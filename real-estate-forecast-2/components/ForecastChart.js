"use client"
import React from "react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts"
import PropTypes from "prop-types"

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="custom-tooltip bg-[#111111] border border-zinc-800 p-3 rounded-md shadow-lg">
        <p className="font-medium text-white">{`Year: ${label}`}</p>
        <p className="text-cyan-400">{`Price: $${payload[0].value?.toLocaleString()}`}</p>
      </div>
    )
  }
  return null
}

const ForecastChart = ({ years, prices, metro }) => {
  if (!years || !prices || years.length !== prices.length) {
    console.error("Invalid data for ForecastChart:", { years, prices, metro });
    return null;
  }

  // Combine years and prices into a single array of objects
  const data = years.map((year, index) => ({
    year,
    price: prices[index],
  }));

  if (data.length === 0) {
    console.warn("No data available to render the chart.");
    return null;
  }

  // Debugging: Check if the data is being correctly passed to the chart
  console.log("Data passed to ForecastChart:", data);

  // Debugging: Check if the component is being rendered
  console.log("ForecastChart component is rendering.");

  console.log("years:", years);
  console.log("prices:", prices);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart
        data={data}
        margin={{
          top: 10,
          right: 30,
          left: 20,
          bottom: 30,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
        <XAxis
          dataKey="year"
          tick={{ fill: "#a3a3a3" }}
          axisLine={{ stroke: "#333333" }}
          tickLine={{ stroke: "#333333" }}
        />
        <YAxis
          tickFormatter={(value) => `$${value.toLocaleString()}`}
          tick={{ fill: "#a3a3a3" }}
          axisLine={{ stroke: "#333333" }}
          tickLine={{ stroke: "#333333" }}
        />
        <Tooltip content={<CustomTooltip />} />
        <Line
          type="monotone"
          dataKey="price"
          stroke="#22d3ee"
          strokeWidth={2}
          dot={{ fill: "#22d3ee", r: 4 }}
          activeDot={{ r: 6, fill: "#06b6d4" }}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

ForecastChart.propTypes = {
  years: PropTypes.arrayOf(PropTypes.number).isRequired,
  prices: PropTypes.arrayOf(PropTypes.number).isRequired,
  metro: PropTypes.string.isRequired,
}

export default ForecastChart