import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Connect 4 - AlphaZero AI',
  description: 'Play Connect 4 against an AlphaZero-style AI in your browser',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body style={{ margin: 0, backgroundColor: '#f3f4f6', minHeight: '100vh' }}>
        {children}
      </body>
    </html>
  );
}
