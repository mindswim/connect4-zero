/**
 * Connect 4 game logic for browser.
 * Mirrors the Python implementation exactly.
 */

export const ROWS = 6;
export const COLS = 7;
export const NUM_CHANNELS = 2;

export type Board = number[][];  // 6x7 board, 0=empty, 1=player1, 2=player2
export type Player = 1 | 2;

export function createBoard(): Board {
  return Array(ROWS).fill(null).map(() => Array(COLS).fill(0));
}

export function copyBoard(board: Board): Board {
  return board.map(row => [...row]);
}

export function getLegalMoves(board: Board): boolean[] {
  // A column is legal if the top row is empty
  return board[0].map(cell => cell === 0);
}

export function applyMove(board: Board, col: number, player: Player): Board {
  const newBoard = copyBoard(board);
  // Drop piece to lowest empty row
  for (let row = ROWS - 1; row >= 0; row--) {
    if (newBoard[row][col] === 0) {
      newBoard[row][col] = player;
      break;
    }
  }
  return newBoard;
}

export function checkWin(board: Board, player: Player): boolean {
  // Check horizontal
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col <= COLS - 4; col++) {
      if (
        board[row][col] === player &&
        board[row][col + 1] === player &&
        board[row][col + 2] === player &&
        board[row][col + 3] === player
      ) {
        return true;
      }
    }
  }

  // Check vertical
  for (let row = 0; row <= ROWS - 4; row++) {
    for (let col = 0; col < COLS; col++) {
      if (
        board[row][col] === player &&
        board[row + 1][col] === player &&
        board[row + 2][col] === player &&
        board[row + 3][col] === player
      ) {
        return true;
      }
    }
  }

  // Check diagonal (down-right)
  for (let row = 0; row <= ROWS - 4; row++) {
    for (let col = 0; col <= COLS - 4; col++) {
      if (
        board[row][col] === player &&
        board[row + 1][col + 1] === player &&
        board[row + 2][col + 2] === player &&
        board[row + 3][col + 3] === player
      ) {
        return true;
      }
    }
  }

  // Check diagonal (down-left)
  for (let row = 0; row <= ROWS - 4; row++) {
    for (let col = 3; col < COLS; col++) {
      if (
        board[row][col] === player &&
        board[row + 1][col - 1] === player &&
        board[row + 2][col - 2] === player &&
        board[row + 3][col - 3] === player
      ) {
        return true;
      }
    }
  }

  return false;
}

export function isBoardFull(board: Board): boolean {
  return board[0].every(cell => cell !== 0);
}

export function isTerminal(board: Board): { done: boolean; winner: Player | null } {
  if (checkWin(board, 1)) return { done: true, winner: 1 };
  if (checkWin(board, 2)) return { done: true, winner: 2 };
  if (isBoardFull(board)) return { done: true, winner: null };
  return { done: false, winner: null };
}

/**
 * Encode board to tensor format for neural network.
 * Returns [2, 6, 7] array in canonical form (from current player's perspective).
 */
export function encodeBoard(board: Board, currentPlayer: Player): Float32Array {
  const tensor = new Float32Array(NUM_CHANNELS * ROWS * COLS);

  const myPlayer = currentPlayer;
  const theirPlayer = currentPlayer === 1 ? 2 : 1;

  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLS; col++) {
      const cell = board[row][col];
      const idx = row * COLS + col;

      // Channel 0: my pieces
      if (cell === myPlayer) {
        tensor[idx] = 1.0;
      }
      // Channel 1: opponent pieces
      if (cell === theirPlayer) {
        tensor[ROWS * COLS + idx] = 1.0;
      }
    }
  }

  return tensor;
}
