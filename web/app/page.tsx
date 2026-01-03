'use client';

import Connect4Game from '../components/Connect4Game';

export default function Home() {
  return (
    <main style={styles.main}>
      <div style={styles.container}>
        <div style={styles.header}>
          <h1 style={styles.title}>Connect 4</h1>
          <p style={styles.subtitle}>Play against an AlphaZero-trained AI</p>
        </div>

        <div style={styles.gameArea}>
          <Connect4Game
            modelPath="/model.onnx"
            numSimulations={100}
            playerFirst={true}
          />
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
  gameArea: {
    display: 'flex',
    justifyContent: 'center',
  },
};
