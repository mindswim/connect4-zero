'use client';

import { useState, useEffect, useCallback } from 'react';
import {
  Board,
  Player,
  ROWS,
  COLS,
  createBoard,
  getLegalMoves,
  applyMove,
  isTerminal,
} from '../lib/game';
import { Connect4AI, RandomAI } from '../lib/ai';

type GameState = 'loading' | 'ready' | 'playing' | 'thinking' | 'previewing' | 'gameover';
type Winner = 'player' | 'ai' | 'draw' | null;

interface Props {
  modelPath?: string;
  numSimulations?: number;
  playerFirst?: boolean;
}

export default function Connect4Game({
  modelPath = '/model.onnx',
  numSimulations = 50,
  playerFirst = true,
}: Props) {
  const [board, setBoard] = useState<Board>(createBoard);
  const [currentPlayer, setCurrentPlayer] = useState<Player>(1);
  const [gameState, setGameState] = useState<GameState>('loading');
  const [winner, setWinner] = useState<Winner>(null);
  const [ai, setAi] = useState<Connect4AI | RandomAI | null>(null);
  const [hoverCol, setHoverCol] = useState<number | null>(null);
  const [lastMove, setLastMove] = useState<{ row: number; col: number } | null>(null);
  const [moveHistory, setMoveHistory] = useState<number[]>([]);
  const [useRandom, setUseRandom] = useState(false);
  const [aiPreviewCol, setAiPreviewCol] = useState<number | null>(null);

  const humanPlayer: Player = playerFirst ? 1 : 2;
  const aiPlayer: Player = playerFirst ? 2 : 1;

  // Load model on mount
  useEffect(() => {
    let mounted = true;

    async function loadAI() {
      try {
        const aiInstance = new Connect4AI(numSimulations);
        await aiInstance.loadModel(modelPath);
        if (mounted) {
          setAi(aiInstance);
          setGameState('ready');
        }
      } catch (error) {
        console.warn('Failed to load model, using random AI:', error);
        if (mounted) {
          setAi(new RandomAI());
          setUseRandom(true);
          setGameState('ready');
        }
      }
    }

    loadAI();

    return () => {
      mounted = false;
    };
  }, [modelPath, numSimulations]);

  // AI move
  const makeAIMove = useCallback(async () => {
    if (!ai || gameState !== 'playing') return;

    setGameState('thinking');

    let col: number;
    if (ai instanceof Connect4AI) {
      col = await ai.getBestMove(board, currentPlayer);
    } else {
      col = ai.getBestMove(board, currentPlayer);
    }

    // Show preview at top of column
    setAiPreviewCol(col);
    setGameState('previewing');

    // Wait a moment so user can see the preview
    await new Promise(resolve => setTimeout(resolve, 400));

    // Find landing row
    let landingRow = 0;
    for (let row = ROWS - 1; row >= 0; row--) {
      if (board[row][col] === 0) {
        landingRow = row;
        break;
      }
    }

    // Place the piece
    const newBoard = applyMove(board, col, currentPlayer);
    setBoard(newBoard);
    setLastMove({ row: landingRow, col });
    setMoveHistory(prev => [...prev, col]);
    setAiPreviewCol(null);

    const { done, winner: gameWinner } = isTerminal(newBoard);
    if (done) {
      setGameState('gameover');
      if (gameWinner === humanPlayer) {
        setWinner('player');
      } else if (gameWinner === aiPlayer) {
        setWinner('ai');
      } else {
        setWinner('draw');
      }
    } else {
      setCurrentPlayer(currentPlayer === 1 ? 2 : 1);
      setGameState('playing');
    }
  }, [ai, board, currentPlayer, gameState, humanPlayer, aiPlayer]);

  // Handle human move
  const handleColumnClick = useCallback((col: number) => {
    if (gameState !== 'playing' || currentPlayer !== humanPlayer) return;

    const legalMoves = getLegalMoves(board);
    if (!legalMoves[col]) return;

    // Find landing row
    let landingRow = 0;
    for (let row = ROWS - 1; row >= 0; row--) {
      if (board[row][col] === 0) {
        landingRow = row;
        break;
      }
    }

    const newBoard = applyMove(board, col, currentPlayer);
    setBoard(newBoard);
    setLastMove({ row: landingRow, col });
    setMoveHistory(prev => [...prev, col]);

    const { done, winner: gameWinner } = isTerminal(newBoard);
    if (done) {
      setGameState('gameover');
      if (gameWinner === humanPlayer) {
        setWinner('player');
      } else if (gameWinner === aiPlayer) {
        setWinner('ai');
      } else {
        setWinner('draw');
      }
    } else {
      setCurrentPlayer(currentPlayer === 1 ? 2 : 1);
    }
  }, [board, currentPlayer, gameState, humanPlayer, aiPlayer]);

  // Trigger AI move when it's AI's turn
  useEffect(() => {
    if (gameState === 'playing' && currentPlayer === aiPlayer) {
      makeAIMove();
    }
  }, [gameState, currentPlayer, aiPlayer, makeAIMove]);

  const startGame = () => {
    setBoard(createBoard());
    setCurrentPlayer(1);
    setWinner(null);
    setLastMove(null);
    setMoveHistory([]);
    setAiPreviewCol(null);
    setGameState('playing');
  };

  const resetGame = () => {
    setBoard(createBoard());
    setCurrentPlayer(1);
    setWinner(null);
    setLastMove(null);
    setMoveHistory([]);
    setAiPreviewCol(null);
    setGameState('ready');
  };

  const legalMoves = getLegalMoves(board);
  const isHumanTurn = gameState === 'playing' && currentPlayer === humanPlayer;

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Connect 4</h1>

      {useRandom && (
        <div style={styles.warning}>
          Model not found. Playing against random AI.
        </div>
      )}

      <div style={styles.status}>
        {gameState === 'loading' && 'Loading AI...'}
        {gameState === 'ready' && 'Click Start to begin!'}
        {gameState === 'playing' && isHumanTurn && 'Your turn - click a column'}
        {(gameState === 'thinking' || gameState === 'previewing') && 'AI is thinking...'}
        {gameState === 'gameover' && winner === 'player' && 'You win!'}
        {gameState === 'gameover' && winner === 'ai' && 'AI wins!'}
        {gameState === 'gameover' && winner === 'draw' && 'Draw!'}
      </div>

      <div style={styles.boardContainer}>
        {/* Hover/Preview indicator row */}
        <div style={styles.hoverRow}>
          <div style={{ width: 8 }} />
          {Array(COLS).fill(null).map((_, col) => {
            const showHumanHover = isHumanTurn && legalMoves[col] && hoverCol === col;
            const showAiPreview = aiPreviewCol === col;

            return (
              <div key={col} style={styles.cell}>
                <div
                  style={{
                    ...styles.piece,
                    ...(showAiPreview ? styles.player2 : styles.player1),
                    opacity: showHumanHover || showAiPreview ? 1 : 0,
                    transition: 'opacity 0.15s',
                  }}
                />
              </div>
            );
          })}
        </div>

        {/* Main board */}
        <div
          style={styles.grid}
          onMouseLeave={() => setHoverCol(null)}
        >
          {board.map((row, rowIdx) => (
            <div key={rowIdx} style={styles.row}>
              {row.map((cell, colIdx) => (
                <div
                  key={colIdx}
                  style={{
                    ...styles.cell,
                    cursor: isHumanTurn && legalMoves[colIdx] ? 'pointer' : 'default',
                  }}
                  onClick={() => handleColumnClick(colIdx)}
                  onMouseEnter={() => setHoverCol(colIdx)}
                >
                  <div
                    style={{
                      ...styles.piece,
                      ...(cell === 1 ? styles.player1 : cell === 2 ? styles.player2 : styles.empty),
                      ...(lastMove?.row === rowIdx && lastMove?.col === colIdx ? styles.lastMove : {}),
                    }}
                  />
                </div>
              ))}
            </div>
          ))}
        </div>

        {/* Column numbers */}
        <div style={styles.columnNumbers}>
          <div style={{ width: 8 }} />
          {Array(COLS).fill(null).map((_, col) => (
            <div key={col} style={styles.columnNumber}>{col}</div>
          ))}
        </div>
      </div>

      <div style={styles.controls}>
        {gameState === 'ready' && (
          <button onClick={startGame} style={styles.button}>
            Start Game
          </button>
        )}
        {(gameState === 'playing' || gameState === 'thinking' || gameState === 'previewing' || gameState === 'gameover') && (
          <button onClick={resetGame} style={styles.button}>
            New Game
          </button>
        )}
      </div>

      {moveHistory.length > 0 && (
        <div style={styles.history}>
          Moves: {moveHistory.join(', ')}
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '20px',
    fontFamily: 'system-ui, sans-serif',
  },
  title: {
    fontSize: '2rem',
    marginBottom: '10px',
  },
  warning: {
    color: '#f59e0b',
    marginBottom: '10px',
    padding: '8px 16px',
    backgroundColor: '#fef3c7',
    borderRadius: '4px',
  },
  status: {
    fontSize: '1.2rem',
    marginBottom: '20px',
    minHeight: '30px',
  },
  boardContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  hoverRow: {
    display: 'flex',
    marginBottom: '4px',
  },
  grid: {
    backgroundColor: '#1d4ed8',
    padding: '8px',
    borderRadius: '8px',
  },
  row: {
    display: 'flex',
  },
  cell: {
    width: '60px',
    height: '60px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '4px',
  },
  piece: {
    width: '48px',
    height: '48px',
    borderRadius: '50%',
    transition: 'all 0.2s',
  },
  empty: {
    backgroundColor: '#172554',
  },
  player1: {
    backgroundColor: '#ef4444',
    boxShadow: 'inset 0 -3px 6px rgba(0,0,0,0.3)',
  },
  player2: {
    backgroundColor: '#fbbf24',
    boxShadow: 'inset 0 -3px 6px rgba(0,0,0,0.3)',
  },
  lastMove: {
    boxShadow: '0 0 0 3px white, inset 0 -3px 6px rgba(0,0,0,0.3)',
  },
  columnNumbers: {
    display: 'flex',
    marginTop: '8px',
  },
  columnNumber: {
    width: '60px',
    textAlign: 'center',
    color: '#6b7280',
    fontSize: '0.9rem',
  },
  controls: {
    marginTop: '20px',
    display: 'flex',
    gap: '10px',
  },
  button: {
    padding: '12px 24px',
    fontSize: '1rem',
    backgroundColor: '#1d4ed8',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
  },
  history: {
    marginTop: '20px',
    color: '#6b7280',
    fontSize: '0.9rem',
  },
};
