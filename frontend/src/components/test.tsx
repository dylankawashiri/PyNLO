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
import { Button, Card, Grid, TextField, Typography, Checkbox } from '@mui/material';
import NumberField from './NumberField';

type DudleyStep = 'Initialize' | 'Propagate' | 'Analyze' | 'Done';

type DudleyEvent = {
    step: DudleyStep;
    message: string;
    estimated_time_remaining?: number;
    wl?: number[];
    d?: number[];
    beta?: number[];
    img1?: string;
};

type Settings = {
    dz: number;
    steps: number;
    centerWl: number;
    fiberLength: number;
    pumpPower: number;
    pumpPulseLength: number;
    nPoints: number;
}

export default function Test() {
    const [timeLeft, setTimeLeft] = useState(0);
    const [wl, setWl] = useState<number[]>([]);
    const [d, setD] = useState<number[]>([]);
    const [beta, setBeta] = useState<number[]>([]);
    const [status, setStatus] = useState('Idle');
    const [enable, setEnable] = useState(false);
    const [img1Data, setImg1Data] = useState<string | null>(null);
    const [settings, setSettings] = useState<Settings>({dz: 1e-3, steps: 100, centerWl: 835.0, fiberLength: 0.15, pumpPower: 1.0e4, pumpPulseLength: 28.4e-3, nPoints: 2**13});
    const [running, setRunning] = useState(false);

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

                setRunning(true);

                setStatus(data.message ?? data.step);

                if (data.step === "Propagate") {
                    setTimeLeft(data.estimated_time_remaining ?? 0);
                } else if (data.step === "Analyze") {
                    setWl(data.wl ?? []);
                    setD(data.d ?? []);
                    setBeta(data.beta ?? []);
                    setImg1Data(`data:image/png;base64,${data.img1 ?? ''}`);

                } else if (data.step === 'Done') {
                    source.close();
                    setEnable(false);
                    setRunning(false);
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

    // Sync settings to server only on initial load
    useEffect(() => {
        fetch("http://localhost:2048/api/dudley_ssfm/settings", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(settings),
        })
        .catch(error => console.log(error));
    }, []); // Empty dependency array - runs once on mount

    // Debounced sync when settings change from NumberFields
    useEffect(() => {
        const timer = setTimeout(() => {
            fetch("http://localhost:2048/api/dudley_ssfm/settings", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(settings),
            })
            .catch(error => console.log(error));
        }, 500); // Wait 500ms after last change before syncing

        return () => clearTimeout(timer);
    }, [settings]);

    const handleChange = (setting: string) => (value: number | null) => {
        if (value !== null && running === false) {
            setSettings((prevSettings) => ({
                ...prevSettings,
                [setting]: value,
            }));
        }
    };

    const option = {
        xAxis: { data: wl },
        yAxis: { type: 'value' },
        series: [{ data: d, type: 'line' }],
    };

    return (
        <Grid container spacing={2}>
            <Grid size={12}>
                <Card>
                <Typography variant='h4'>
                    <b>Dudley SSFM Simulation</b>
                </Typography>
                <Button variant="contained" onClick={() => setEnable(!enable)}>Start</Button>
                <p>Status: {status}</p>
                <p>Time left: {timeLeft.toFixed(2)} seconds</p>
                <Grid container rowSpacing={2} columnSpacing={2}>
                    {Object.entries(settings).map(([setting, value]) => 
                        <Grid size={3}>
                            <NumberField
                                key={setting}
                                label={setting}
                                defaultValue={value}
                                onValueChange={handleChange(setting)}
                            />
                        </Grid>
                    )}
                </Grid>
                <Grid size={12}>
                    <Card>
                        {img1Data && <img src={img1Data} alt="Propagation Wavelength" />}
                    </Card>
                </Grid>
            </Card>
            </Grid>  
        </Grid>
    );
}
