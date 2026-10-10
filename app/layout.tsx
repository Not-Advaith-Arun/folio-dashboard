import type { Metadata } from "next";
import "@fontsource-variable/geist";
import "react-grid-layout/css/styles.css";
import "react-resizable/css/styles.css";
import "./globals.css";
import "./design.css";

export const metadata: Metadata = {
  title: "Folio · Your portfolio, clearly",
  description: "A private, free-first portfolio dashboard for stocks and mutual funds.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
