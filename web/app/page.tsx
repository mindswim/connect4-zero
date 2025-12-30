import Connect4Game from '../components/Connect4Game';

export default function Home() {
  return (
    <main style={{
      display: 'flex',
      justifyContent: 'center',
      paddingTop: '40px',
      minHeight: '100vh',
    }}>
      <Connect4Game
        modelPath="/model.onnx"
        numSimulations={50}
        playerFirst={true}
      />
    </main>
  );
}
