'use client';

import { useState, useCallback } from 'react';
import Connect4Game from '../components/Connect4Game';
import DifficultySlider, { DifficultyPresets, getDifficultyConfig, DifficultyConfig } from '../components/DifficultySlider';

export default function Home() {
  const [difficulty, setDifficulty] = useState(45);
  const [difficultyConfig, setDifficultyConfig] = useState<DifficultyConfig>(
    getDifficultyConfig(45)
  );
  const [gameKey, setGameKey] = useState(0);
  const [playerFirst, setPlayerFirst] = useState(true);

  const handleDifficultyChange = useCallback((value: number, config: DifficultyConfig) => {
    setDifficulty(value);
    setDifficultyConfig(config);
  }, []);

  const handleNewGame = useCallback(() => {
    setGameKey(prev => prev + 1);
  }, []);

  return (
    <main style={styles.main}>
      <div style={styles.container}>
        {/* Header */}
        <div style={styles.header}>
          <h1 style={styles.title}>Connect 4</h1>
          <p style={styles.subtitle}>Play against an AlphaZero-trained AI</p>
        </div>

        {/* Settings Panel */}
        <div style={styles.settings}>
          <DifficultySlider
            value={difficulty}
            onChange={handleDifficultyChange}
          />
          <DifficultyPresets
            onSelect={handleDifficultyChange}
            currentValue={difficulty}
          />

          {/* Player Order */}
          <div style={styles.playerOrder}>
            <span style={styles.orderLabel}>Play as:</span>
            <div style={styles.orderButtons}>
              <button
                onClick={() => setPlayerFirst(true)}
                style={{
                  ...styles.orderButton,
                  ...(playerFirst ? styles.orderButtonActive : {}),
                }}
              >
                First (Red)
              </button>
              <button
                onClick={() => setPlayerFirst(false)}
                style={{
                  ...styles.orderButton,
                  ...(!playerFirst ? styles.orderButtonActive : {}),
                }}
              >
                Second (Yellow)
              </button>
            </div>
          </div>

          <button onClick={handleNewGame} style={styles.newGameButton}>
            New Game
          </button>
        </div>

        {/* Game Area */}
        <div style={styles.gameArea}>
          <Connect4Game
            key={gameKey}
            modelPath="/model.onnx"
            numSimulations={difficultyConfig.simulations}
            playerFirst={playerFirst}
          />
        </div>

        {/* Footer */}
        <div style={styles.footer}>
          <p>
            AI trained using the <strong>AlphaZero</strong> algorithm
          </p>
          <p style={styles.footerSub}>
            {difficultyConfig.name}: {difficultyConfig.simulations} MCTS simulations per move
          </p>
        </div>
      </div>
    </main>
  );
}

const styles: Record<string, React.CSSProperties> = {
  main: {
    minHeight: '100vh',
    backgroundColor: '#f9fafb',
    padding: '20px',
  },
  container: {
    maxWidth: '800px',
    margin: '0 auto',
  },
  header: {
    textAlign: 'center',
    marginBottom: '24px',
  },
  title: {
    fontSize: '2.5rem',
    fontWeight: '700',
    color: '#1f2937',
    marginBottom: '8px',
  },
  subtitle: {
    fontSize: '1.1rem',
    color: '#6b7280',
  },
  settings: {
    backgroundColor: 'white',
    padding: '20px',
    borderRadius: '12px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
    marginBottom: '24px',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '16px',
  },
  playerOrder: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    marginTop: '8px',
  },
  orderLabel: {
    fontSize: '0.875rem',
    color: '#6b7280',
  },
  orderButtons: {
    display: 'flex',
    gap: '8px',
  },
  orderButton: {
    padding: '8px 16px',
    fontSize: '0.875rem',
    backgroundColor: '#f3f4f6',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
  },
  orderButtonActive: {
    backgroundColor: '#1d4ed8',
    color: 'white',
  },
  newGameButton: {
    padding: '12px 24px',
    fontSize: '1rem',
    backgroundColor: '#1d4ed8',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontWeight: '500',
    marginTop: '8px',
  },
  gameArea: {
    display: 'flex',
    justifyContent: 'center',
  },
  footer: {
    textAlign: 'center',
    marginTop: '32px',
    padding: '20px',
    color: '#6b7280',
    fontSize: '0.9rem',
  },
  footerSub: {
    fontSize: '0.8rem',
    marginTop: '4px',
  },
};
