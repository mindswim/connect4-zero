'use client';

import { useState, useCallback } from 'react';
import Connect4Game from '../components/Connect4Game';
import DifficultySlider, { DifficultyPresets, getDifficultyConfig, DifficultyConfig } from '../components/DifficultySlider';

type GameType = 'connect4' | 'tictactoe';

interface GameInfo {
  name: string;
  description: string;
  modelPath: string;
  available: boolean;
}

const GAMES: Record<GameType, GameInfo> = {
  connect4: {
    name: 'Connect 4',
    description: 'Drop pieces to connect 4 in a row',
    modelPath: '/model.onnx',
    available: true,
  },
  tictactoe: {
    name: 'Tic-Tac-Toe',
    description: 'Classic 3x3 game',
    modelPath: '/tictactoe.onnx',
    available: false, // Coming soon
  },
};

export default function Home() {
  const [selectedGame, setSelectedGame] = useState<GameType>('connect4');
  const [difficulty, setDifficulty] = useState(45); // Medium by default
  const [difficultyConfig, setDifficultyConfig] = useState<DifficultyConfig>(
    getDifficultyConfig(45)
  );
  const [gameKey, setGameKey] = useState(0); // For forcing re-render
  const [playerFirst, setPlayerFirst] = useState(true);

  const handleDifficultyChange = useCallback((value: number, config: DifficultyConfig) => {
    setDifficulty(value);
    setDifficultyConfig(config);
  }, []);

  const handleNewGame = useCallback(() => {
    setGameKey(prev => prev + 1);
  }, []);

  const gameInfo = GAMES[selectedGame];

  return (
    <main style={styles.main}>
      <div style={styles.container}>
        {/* Header */}
        <div style={styles.header}>
          <h1 style={styles.title}>AI Game Arcade</h1>
          <p style={styles.subtitle}>Play against AlphaZero-trained AI</p>
        </div>

        {/* Game Selector */}
        <div style={styles.gameSelector}>
          {Object.entries(GAMES).map(([key, game]) => (
            <button
              key={key}
              onClick={() => game.available && setSelectedGame(key as GameType)}
              style={{
                ...styles.gameButton,
                ...(selectedGame === key ? styles.gameButtonActive : {}),
                ...(game.available ? {} : styles.gameButtonDisabled),
              }}
              disabled={!game.available}
            >
              <span style={styles.gameName}>{game.name}</span>
              {!game.available && <span style={styles.comingSoon}>Soon</span>}
            </button>
          ))}
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
            New Game with Settings
          </button>
        </div>

        {/* Game Area */}
        <div style={styles.gameArea}>
          {selectedGame === 'connect4' && (
            <Connect4Game
              key={gameKey}
              modelPath={gameInfo.modelPath}
              numSimulations={difficultyConfig.simulations}
              playerFirst={playerFirst}
            />
          )}
        </div>

        {/* Footer */}
        <div style={styles.footer}>
          <p>
            AI trained using <strong>AlphaZero</strong> algorithm
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
  gameSelector: {
    display: 'flex',
    gap: '12px',
    justifyContent: 'center',
    marginBottom: '24px',
  },
  gameButton: {
    padding: '16px 24px',
    fontSize: '1rem',
    backgroundColor: 'white',
    border: '2px solid #e5e7eb',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'all 0.15s',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '4px',
  },
  gameButtonActive: {
    borderColor: '#1d4ed8',
    backgroundColor: '#eff6ff',
  },
  gameButtonDisabled: {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
  gameName: {
    fontWeight: '600',
  },
  comingSoon: {
    fontSize: '0.75rem',
    color: '#9ca3af',
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
