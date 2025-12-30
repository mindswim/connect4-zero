/**
 * AI player using ONNX model for inference.
 */

import * as ort from 'onnxruntime-web';
import { Board, Player, ROWS, COLS, NUM_CHANNELS, encodeBoard } from './game';
import { MCTS, MCTSNode } from './mcts';

// Configure ONNX Runtime
ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'error';  // Suppress warnings

export class Connect4AI {
  private session: ort.InferenceSession | null = null;
  private mcts: MCTS | null = null;
  private numSimulations: number;

  constructor(numSimulations: number = 50) {
    this.numSimulations = numSimulations;
  }

  /**
   * Load the ONNX model from a URL or path.
   */
  async loadModel(modelPath: string): Promise<void> {
    // Fetch the model as ArrayBuffer for more reliable loading
    const response = await fetch(modelPath);
    if (!response.ok) {
      throw new Error(`Failed to fetch model: ${response.status}`);
    }
    const modelBuffer = await response.arrayBuffer();

    this.session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['wasm'],
    });

    // Create MCTS with our evaluate function
    this.mcts = new MCTS(
      (board: Board, player: Player) => this.evaluate(board, player),
      1.5  // c_puct
    );

    console.log('Model loaded successfully');
  }

  /**
   * Evaluate a position using the neural network.
   */
  private async evaluate(board: Board, player: Player): Promise<{ policy: Float32Array; value: number }> {
    if (!this.session) {
      throw new Error('Model not loaded');
    }

    // Encode board
    const inputData = encodeBoard(board, player);

    // Create tensor [1, 2, 6, 7]
    const inputTensor = new ort.Tensor('float32', inputData, [1, NUM_CHANNELS, ROWS, COLS]);

    // Run inference
    const results = await this.session.run({ board: inputTensor });

    // Get outputs
    const policyOutput = results['policy'];
    const valueOutput = results['value'];

    // Apply softmax to policy
    const policyData = policyOutput.data as Float32Array;
    const policy = this.softmax(policyData);

    // Get value scalar
    const value = (valueOutput.data as Float32Array)[0];

    return { policy, value };
  }

  /**
   * Softmax function.
   */
  private softmax(logits: Float32Array): Float32Array {
    const maxLogit = Math.max(...logits);
    const expLogits = new Float32Array(logits.length);
    let sumExp = 0;

    for (let i = 0; i < logits.length; i++) {
      expLogits[i] = Math.exp(logits[i] - maxLogit);
      sumExp += expLogits[i];
    }

    for (let i = 0; i < logits.length; i++) {
      expLogits[i] /= sumExp;
    }

    return expLogits;
  }

  /**
   * Get the best move for the current position.
   */
  async getBestMove(board: Board, player: Player): Promise<number> {
    if (!this.mcts) {
      throw new Error('Model not loaded');
    }

    const root = await this.mcts.search(board, player, this.numSimulations);
    return MCTS.selectAction(root);
  }

  /**
   * Get move with policy distribution (for visualization).
   */
  async getMoveWithPolicy(board: Board, player: Player): Promise<{ action: number; policy: Float32Array }> {
    if (!this.mcts) {
      throw new Error('Model not loaded');
    }

    const root = await this.mcts.search(board, player, this.numSimulations);
    const action = MCTS.selectAction(root);
    const policy = MCTS.getPolicy(root, 0);

    return { action, policy };
  }

  /**
   * Check if model is loaded.
   */
  isLoaded(): boolean {
    return this.session !== null;
  }
}

/**
 * Random AI for testing without model.
 */
export class RandomAI {
  getBestMove(board: Board, _player: Player): number {
    const legalMoves: number[] = [];
    for (let col = 0; col < COLS; col++) {
      if (board[0][col] === 0) {
        legalMoves.push(col);
      }
    }
    return legalMoves[Math.floor(Math.random() * legalMoves.length)];
  }
}
