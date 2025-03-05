import { NextResponse } from 'next/server';

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const state = searchParams.get('state');
    const metro = searchParams.get('metro');
    const date = searchParams.get('date');

    if (!state || !metro || !date) {
      return NextResponse.json(
        { error: 'State, metro, and date parameters are required' },
        { status: 400 }
      );
    }

    const response = await fetch(
      `http://127.0.0.1:8050/api/chart-data?state=${state}&metro=${metro}&date=${date}`
    );
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching chart data:', error);
    return NextResponse.json({ error: 'Failed to fetch chart data' }, { status: 500 });
  }
} 