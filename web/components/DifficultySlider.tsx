'use client';

import { useState, useCallback } from 'react';

interface DifficultyConfig {
  simulations: number;
  temperature: number;
  name: string;
  description: string;
}

interface Props {
  value: number;
  onChange: (value: number, config: DifficultyConfig) => void;
  disabled?: boolean;
}

// Map slider value (0-100) to difficulty config
function getDifficultyConfig(value: number): DifficultyConfig {
  // Exponential scaling for simulations
  const minSims = 10;
  const maxSims = 500;
  const logMin = Math.log(minSims);
  const logMax = Math.log(maxSims);
  const t = value / 100;
  const simulations = Math.round(Math.exp(logMin + t * (logMax - logMin)));

  // Temperature decreases as difficulty increases
  const temperature = Math.max(0, 1 - t);

  // Name based on value
  let name: string;
  let description: string;
  if (value < 20) {
    name = 'Beginner';
    description = 'Makes frequent mistakes';
  } else if (value < 40) {
    name = 'Easy';
    description = 'Casual play';
  } else if (value < 60) {
    name = 'Medium';
    description = 'Balanced challenge';
  } else if (value < 80) {
    name = 'Hard';
    description = 'Strong play';
  } else if (value < 95) {
    name = 'Expert';
    description = 'Very difficult';
  } else {
    name = 'Maximum';
    description = 'Best possible play';
  }

  return { simulations, temperature, name, description };
}

export default function DifficultySlider({ value, onChange, disabled = false }: Props) {
  const config = getDifficultyConfig(value);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseInt(e.target.value, 10);
    onChange(newValue, getDifficultyConfig(newValue));
  }, [onChange]);

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.label}>Difficulty</span>
        <span style={styles.name}>{config.name}</span>
      </div>

      <input
        type="range"
        min="0"
        max="100"
        value={value}
        onChange={handleChange}
        disabled={disabled}
        style={styles.slider}
      />

      <div style={styles.footer}>
        <span style={styles.description}>{config.description}</span>
        <span style={styles.sims}>{config.simulations} thinking steps</span>
      </div>
    </div>
  );
}

// Preset buttons component
interface PresetProps {
  onSelect: (value: number, config: DifficultyConfig) => void;
  currentValue: number;
  disabled?: boolean;
}

export function DifficultyPresets({ onSelect, currentValue, disabled = false }: PresetProps) {
  const presets = [
    { value: 15, label: 'Easy' },
    { value: 45, label: 'Medium' },
    { value: 75, label: 'Hard' },
    { value: 100, label: 'Max' },
  ];

  return (
    <div style={styles.presets}>
      {presets.map(preset => (
        <button
          key={preset.value}
          onClick={() => onSelect(preset.value, getDifficultyConfig(preset.value))}
          disabled={disabled}
          style={{
            ...styles.presetButton,
            ...(Math.abs(currentValue - preset.value) < 10 ? styles.presetActive : {}),
          }}
        >
          {preset.label}
        </button>
      ))}
    </div>
  );
}

export { getDifficultyConfig };
export type { DifficultyConfig };

const styles: Record<string, React.CSSProperties> = {
  container: {
    width: '100%',
    maxWidth: '300px',
    padding: '16px',
    backgroundColor: '#f3f4f6',
    borderRadius: '8px',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '8px',
  },
  label: {
    fontSize: '0.875rem',
    color: '#6b7280',
  },
  name: {
    fontSize: '1rem',
    fontWeight: '600',
    color: '#1f2937',
  },
  slider: {
    width: '100%',
    height: '8px',
    borderRadius: '4px',
    appearance: 'none',
    background: 'linear-gradient(to right, #86efac, #fbbf24, #ef4444)',
    cursor: 'pointer',
  },
  footer: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: '8px',
  },
  description: {
    fontSize: '0.75rem',
    color: '#9ca3af',
  },
  sims: {
    fontSize: '0.75rem',
    color: '#6b7280',
  },
  presets: {
    display: 'flex',
    gap: '8px',
    marginTop: '12px',
  },
  presetButton: {
    flex: 1,
    padding: '8px 12px',
    fontSize: '0.875rem',
    backgroundColor: '#e5e7eb',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    transition: 'all 0.15s',
  },
  presetActive: {
    backgroundColor: '#1d4ed8',
    color: 'white',
  },
};
