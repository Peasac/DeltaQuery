import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "DeltaQuery trace",
  description: "Plan diff and evidence inspection for DeltaQuery sessions",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
