'use client';

import { useState, useEffect } from 'react';
import { Box, Container, Typography, FormControl, InputLabel, Select, MenuItem, Paper, CircularProgress } from '@mui/material';
import dynamic from 'next/dynamic';

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export default function Dashboard() {
  const [states, setStates] = useState<string[]>([]);
  const [metros, setMetros] = useState<string[]>([]);
  const [dates, setDates] = useState<string[]>([]);
  const [selectedState, setSelectedState] = useState<string>('');
  const [selectedMetro, setSelectedMetro] = useState<string>('');
  const [selectedDate, setSelectedDate] = useState<string>('');
  const [chartData, setChartData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Fetch initial data
    fetchStates();
    fetchDates();
  }, []);

  useEffect(() => {
    if (selectedState) {
      fetchMetros(selectedState);
    }
  }, [selectedState]);

  useEffect(() => {
    if (selectedState && selectedMetro && selectedDate) {
      fetchChartData();
    }
  }, [selectedState, selectedMetro, selectedDate]);

  const fetchStates = async () => {
    try {
      const response = await fetch('/api/states');
      const data = await response.json();
      setStates(data.states);
    } catch (error) {
      console.error('Error fetching states:', error);
    }
  };

  const fetchMetros = async (state: string) => {
    try {
      const response = await fetch(`/api/metros?state=${state}`);
      const data = await response.json();
      setMetros(data.metros);
    } catch (error) {
      console.error('Error fetching metros:', error);
    }
  };

  const fetchDates = async () => {
    try {
      const response = await fetch('/api/dates');
      const data = await response.json();
      setDates(data.dates);
      if (data.dates.length > 0) {
        setSelectedDate(data.dates[data.dates.length - 1]);
      }
    } catch (error) {
      console.error('Error fetching dates:', error);
    }
  };

  const fetchChartData = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `/api/chart-data?state=${selectedState}&metro=${selectedMetro}&date=${selectedDate}`
      );
      const data = await response.json();
      setChartData(data);
    } catch (error) {
      console.error('Error fetching chart data:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom align="center">
        Real Estate Market Dashboard
      </Typography>

      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <FormControl sx={{ minWidth: 200 }}>
            <InputLabel>State</InputLabel>
            <Select
              value={selectedState}
              label="State"
              onChange={(e) => setSelectedState(e.target.value)}
            >
              {states.map((state) => (
                <MenuItem key={state} value={state}>
                  {state}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl sx={{ minWidth: 200 }}>
            <InputLabel>Metro Area</InputLabel>
            <Select
              value={selectedMetro}
              label="Metro Area"
              onChange={(e) => setSelectedMetro(e.target.value)}
            >
              {metros.map((metro) => (
                <MenuItem key={metro} value={metro}>
                  {metro}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl sx={{ minWidth: 200 }}>
            <InputLabel>Date</InputLabel>
            <Select
              value={selectedDate}
              label="Date"
              onChange={(e) => setSelectedDate(e.target.value)}
            >
              {dates.map((date) => (
                <MenuItem key={date} value={date}>
                  {date}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
      </Paper>

      <Paper elevation={3} sx={{ p: 3, height: 600 }}>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <CircularProgress />
          </Box>
        ) : chartData ? (
          <Plot
            data={chartData.data}
            layout={chartData.layout}
            config={{ responsive: true }}
            style={{ width: '100%', height: '100%' }}
          />
        ) : (
          <Typography variant="h6" align="center" color="text.secondary">
            Select filters to view the chart
          </Typography>
        )}
      </Paper>
    </Container>
  );
} 