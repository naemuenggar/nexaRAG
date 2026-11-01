import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "nexaRAG",
  description: "Ask questions about your PDFs and get answers with citations.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
