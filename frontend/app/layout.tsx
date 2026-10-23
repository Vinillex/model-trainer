import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Model Trainer",
  description: "Dataset builder and trainer for IMX500 drone detection",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="antialiased">
        <header className="border-b border-neutral-300 dark:border-neutral-800">
          <nav className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3">
            <Link href="/" className="font-semibold">
              Model Trainer
            </Link>
            <span className="text-sm text-neutral-500">IMX500 drone detection</span>
          </nav>
        </header>
        <main className="mx-auto max-w-7xl px-4 py-6">{children}</main>
      </body>
    </html>
  );
}
