import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const response = await fetch('http://127.0.0.1:8050/api/states');
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching states:', error);
    return NextResponse.json({ error: 'Failed to fetch states' }, { status: 500 });
  }
} 