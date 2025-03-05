declare module 'react-plotly.js' {
  import { ComponentType } from 'react';

  interface PlotProps {
    data: any[];
    layout: any;
    config?: any;
    style?: React.CSSProperties;
  }

  const Plot: ComponentType<PlotProps>;
  export default Plot;
} 