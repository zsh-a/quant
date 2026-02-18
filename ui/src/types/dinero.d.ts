declare module 'dinero.js' {
  interface DineroObject {
    getAmount(): number;
    getCurrency(): string;
    toFormat(format?: string, roundingMode?: string): string;
    toUnit(): number;
  }
  function Dinero(options: { amount: number; currency?: string; precision?: number }): DineroObject;
  export default Dinero;
}
