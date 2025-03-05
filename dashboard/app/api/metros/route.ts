import { NextResponse } from 'next/server';

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const state = searchParams.get('state');

    if (!state) {
      return NextResponse.json({ error: 'State parameter is required' }, { status: 400 });
    }

    const response = await fetch(`http://127.0.0.1:8050/api/metros?state=${state}`);
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching metros:', error);
    return NextResponse.json({ error: 'Failed to fetch metros' }, { status: 500 });
  }
} 