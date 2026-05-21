import ReactECharts from 'echarts-for-react';
import { useEffect, useState } from 'react';

import {
  CanvasRenderer,
} from 'echarts/renderers';

import * as echarts from 'echarts/core';

// Register the required components
echarts.use(
  [CanvasRenderer]
);
import { Button } from '@mui/material';

type DudleyStep = 'Initialize' | 'Propagate' | 'Analyze' | 'Done';

type DudleyEvent = {
    step: DudleyStep;
    message: string;
    estimated_time_remaining?: number;
    wl?: number[];
    d?: number[];
    beta?: number[];
    vmin_IW?: number;
    vmax_IW?: number;
    vmin_IT?: number;
    vmax_IT?: number;
    xW?: number[] | Record<string, number>;
    zW?: number[][] | Record<string, number[]>;
    xT?: number[] | Record<string, number>;
    zT?: number[][] | Record<string, number[]>;
    y?: number[] | Record<string, number>;
    img1?: string;
    img2?: string;
};

export default function Test() {
    const [timeLeft, setTimeLeft] = useState(0);
    const [wl, setWl] = useState<number[]>([]);
    const [d, setD] = useState<number[]>([]);
    const [beta, setBeta] = useState<number[]>([]);
    const [status, setStatus] = useState('Idle');
    const [enable, setEnable] = useState(false);
    const [img1Data, setImg1Data] = useState<string | null>(null);
    const [img2Data, setImg2Data] = useState<string | null>(null);

    useEffect(() => {
        if (!enable) {
            return;
        }

        setStatus('Connecting...');
        const source = new EventSource('http://localhost:2048/api/dudley_ssfm');

        source.onmessage = (event) => {
            try {
                const data: DudleyEvent = JSON.parse(event.data);

                if (!data.step) {
                    return;
                }

                setStatus(data.message ?? data.step);

                if (data.step === "Propagate") {
                    setTimeLeft(data.estimated_time_remaining ?? 0);
                } else if (data.step === "Analyze") {
                    setWl(data.wl ?? []);
                    setD(data.d ?? []);
                    setBeta(data.beta ?? []);
                    setImg1Data(`data:image/png;base64,${data.img1 ?? ''}`);
                    setImg2Data(`data:image/png;base64,${data.img2 ?? ''}`);

                } else if (data.step === 'Done') {
                    source.close();
                    setEnable(false);
                }
            } catch (error) {
                setEnable(false);
                console.error("Error fetching data:", error);
            }
        };

        source.onerror = (error) => {
            console.error('SSE connection error:', error);
            setStatus('Connection error');
            source.close();
            setEnable(false);
        };

        return () => {
            source.close();
        };
    }, [enable]);

    const option = {
        xAxis: { data: wl },
        yAxis: { type: 'value' },
        series: [{ data: d, type: 'line' }],
    };

    const option2 = {
        xAxis: { data: wl },
        yAxis: { type: 'value' },
        series: [{ data: beta, type: 'line' }],
    };

    return (
        <div>
            <Button variant="contained" onClick={() => setEnable(!enable)}>Start</Button>
            <p>Status: {status}</p>
            <p>Time left: {timeLeft.toFixed(2)} seconds</p>
            {/* <ReactECharts option={option} />
            <ReactECharts option={option2} /> */}
            {img1Data && <img src={img1Data} alt="Propagation Wavelength" />}

        </div>
    );
}
