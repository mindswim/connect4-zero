/**
 * Monte Carlo Tree Search for browser.
 * Simplified version that works with ONNX model inference.
 */

import {
  Board,
  Player,
  COLS,
  copyBoard,
  getLegalMoves,
  applyMove,
  isTerminal,
  encodeBoard,
} from './game';

export interface MCTSNode {
  board: Board;
  player: Player;
  parent: MCTSNode | null;
  children: Map<number, MCTSNode>;
  action: number | null;
  N: number;  // Visit count
  W: number;  // Total value
  Q: number;  // Mean value (W/N)
  P: number;  // Prior probability
}

function createNode(
  board: Board,
  player: Player,
  parent: MCTSNode | null = null,
  action: number | null = null,
  prior: number = 0
): MCTSNode {
  return {
    board,
    player,
    parent,
    children: new Map(),
    action,
    N: 0,
    W: 0,
    Q: 0,
    P: prior,
  };
}

/**
 * MCTS search using neural network for evaluation.
 */
export class MCTS {
  private evaluateFn: (board: Board, player: Player) => Promise<{ policy: Float32Array; value: number }>;
  private cPuct: number;

  constructor(
    evaluateFn: (board: Board, player: Player) => Promise<{ policy: Float32Array; value: number }>,
    cPuct: number = 1.5
  ) {
    this.evaluateFn = evaluateFn;
    this.cPuct = cPuct;
  }

  /**
   * Run MCTS from the given position.
   */
  async search(board: Board, player: Player, numSimulations: number): Promise<MCTSNode> {
    const root = createNode(board, player);

    // Evaluate root and set priors
    const { policy } = await this.evaluateFn(board, player);
    const legalMoves = getLegalMoves(board);
    this.setRootPriors(root, policy, legalMoves);

    // Run simulations
    for (let i = 0; i < numSimulations; i++) {
      let node = root;

      // Selection: traverse to leaf
      while (node.children.size > 0 && !isTerminal(node.board).done) {
        node = this.selectChild(node);
      }

      // Check if terminal
      const { done, winner } = isTerminal(node.board);

      let value: number;
      if (done) {
        // Terminal node - use actual outcome
        if (winner === null) {
          value = 0;
        } else if (winner === node.player) {
          value = 1;
        } else {
          value = -1;
        }
      } else {
        // Expand and evaluate
        const result = await this.evaluateFn(node.board, node.player);
        value = result.value;
        this.expand(node, result.policy);
      }

      // Backpropagate
      this.backpropagate(node, value);
    }

    return root;
  }

  private setRootPriors(root: MCTSNode, policy: Float32Array, legalMoves: boolean[]): void {
    // Mask illegal moves and renormalize
    let sum = 0;
    for (let i = 0; i < COLS; i++) {
      if (legalMoves[i]) {
        sum += policy[i];
      }
    }

    // Create children with priors
    const nextPlayer = root.player === 1 ? 2 : 1 as Player;
    for (let col = 0; col < COLS; col++) {
      if (legalMoves[col]) {
        const prior = sum > 0 ? policy[col] / sum : 1 / legalMoves.filter(m => m).length;
        const childBoard = applyMove(root.board, col, root.player);
        const child = createNode(childBoard, nextPlayer, root, col, prior);
        root.children.set(col, child);
      }
    }
  }

  private selectChild(node: MCTSNode): MCTSNode {
    let bestChild: MCTSNode | null = null;
    let bestScore = -Infinity;

    const totalN = node.N;

    for (const child of node.children.values()) {
      // PUCT formula
      const u = this.cPuct * child.P * Math.sqrt(totalN) / (1 + child.N);
      const score = -child.Q + u;  // Negate Q because child is opponent

      if (score > bestScore) {
        bestScore = score;
        bestChild = child;
      }
    }

    return bestChild!;
  }

  private expand(node: MCTSNode, policy: Float32Array): void {
    const legalMoves = getLegalMoves(node.board);

    // Mask and renormalize
    let sum = 0;
    for (let i = 0; i < COLS; i++) {
      if (legalMoves[i]) {
        sum += policy[i];
      }
    }

    const nextPlayer = node.player === 1 ? 2 : 1 as Player;
    for (let col = 0; col < COLS; col++) {
      if (legalMoves[col]) {
        const prior = sum > 0 ? policy[col] / sum : 1 / legalMoves.filter(m => m).length;
        const childBoard = applyMove(node.board, col, node.player);
        const child = createNode(childBoard, nextPlayer, node, col, prior);
        node.children.set(col, child);
      }
    }
  }

  private backpropagate(node: MCTSNode, value: number): void {
    let currentNode: MCTSNode | null = node;
    let v = value;

    while (currentNode !== null) {
      currentNode.N += 1;
      currentNode.W += v;
      currentNode.Q = currentNode.W / currentNode.N;
      currentNode = currentNode.parent;
      v = -v;  // Flip value for opponent
    }
  }

  /**
   * Get action probabilities from visit counts.
   */
  static getPolicy(root: MCTSNode, temperature: number = 0): Float32Array {
    const policy = new Float32Array(COLS);

    if (temperature === 0) {
      // Select action with most visits
      let maxN = 0;
      let bestAction = 0;
      for (const [action, child] of root.children) {
        if (child.N > maxN) {
          maxN = child.N;
          bestAction = action;
        }
      }
      policy[bestAction] = 1.0;
    } else {
      // Sample proportionally to visit counts
      let sum = 0;
      for (const child of root.children.values()) {
        sum += Math.pow(child.N, 1 / temperature);
      }
      for (const [action, child] of root.children) {
        policy[action] = Math.pow(child.N, 1 / temperature) / sum;
      }
    }

    return policy;
  }

  /**
   * Select best action (most visits).
   */
  static selectAction(root: MCTSNode): number {
    let maxN = 0;
    let bestAction = 0;
    for (const [action, child] of root.children) {
      if (child.N > maxN) {
        maxN = child.N;
        bestAction = action;
      }
    }
    return bestAction;
  }
}
