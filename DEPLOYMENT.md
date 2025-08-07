# Render Deployment Guide

This guide will help you deploy your Real Estate Valuation Dashboard on Render.

## Prerequisites

1. **Render Account**: Sign up at [render.com](https://render.com)
2. **GitHub Repository**: Your code should be in a GitHub repository
3. **Data Files**: Ensure your CSV files are in the correct locations

## Deployment Steps

### Step 1: Prepare Your Repository

Make sure your repository structure looks like this:
```
├── backend/
│   ├── dashboard/
│   │   └── api.py
│   ├── data/
│   │   └── zillow_housing_cleaned.csv
│   ├── processing/
│   │   └── lstm_metro_forecast_2025_2050.csv
│   └── requirements.txt
├── real-estate-forecast-2/
│   ├── app/
│   │   └── page.tsx
│   ├── components/
│   │   └── ForecastChart.js
│   └── package.json
├── render.yaml
└── README.md
```

### Step 2: Deploy on Render

1. **Log into Render** and click "New +"
2. **Select "Blueprint"** from the dropdown
3. **Connect your GitHub repository**
4. **Render will automatically detect** the `render.yaml` file and deploy both services

### Step 3: Environment Variables

The `render.yaml` file automatically configures:
- **Backend URL**: Automatically set for the frontend
- **Port**: Set to 5002 for the Flask API
- **Python Version**: 3.9.0
- **Node Version**: 18.0.0

### Step 4: Verify Deployment

1. **Backend API**: Should be available at `https://your-api-name.onrender.com`
2. **Frontend Dashboard**: Should be available at `https://your-dashboard-name.onrender.com`

## Troubleshooting

### Common Issues:

1. **Build Failures**:
   - Check that all dependencies are in `requirements.txt`
   - Ensure CSV files are in the correct paths
   - Verify Python/Node versions

2. **API Connection Issues**:
   - Check that the frontend environment variable is set correctly
   - Verify CORS is enabled in the Flask app
   - Test API endpoints directly

3. **Data Loading Issues**:
   - Ensure CSV files are committed to the repository
   - Check file paths in the API code
   - Verify CSV format is correct

### Debugging:

1. **Check Render Logs**: Go to your service dashboard and view logs
2. **Test API Endpoints**: Use curl or Postman to test `/api/metros`
3. **Verify Environment Variables**: Check that `NEXT_PUBLIC_API_BASE_URL` is set

## Local Development

To test locally before deploying:

```bash
# Backend
cd backend/dashboard
python api.py

# Frontend (in another terminal)
cd real-estate-forecast-2
npm run dev
```

## Production Considerations

1. **Data Updates**: The forecast data is static - update by redeploying
2. **Scaling**: Render automatically scales based on traffic
3. **Monitoring**: Use Render's built-in monitoring tools
4. **Custom Domain**: Add your own domain in Render settings

## Cost Optimization

- **Free Tier**: Both services can run on Render's free tier
- **Auto-sleep**: Free services sleep after 15 minutes of inactivity
- **Cold Starts**: First request after sleep may be slower

## Security Notes

- **CORS**: Configured to allow frontend requests
- **Environment Variables**: Sensitive data should be in Render's environment variables
- **HTTPS**: Render provides automatic SSL certificates
