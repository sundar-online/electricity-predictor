/**
 * EB Electricity Billing Dataset — Monthly Time Series
 *
 * NOTE: This is a development dataset generated to match realistic Tamil Nadu
 * residential EB billing patterns. Replace this file with the actual EB
 * billing CSV/JSON when it becomes available.
 *
 * Seasons (Tamil Nadu EB standard):
 *   Summer  : March – June       (high AC/fan usage → high consumption)
 *   Monsoon : July  – October    (moderate usage)
 *   Winter  : November – February (low usage)
 *
 * Features:
 *   month          : 1–12
 *   year           : calendar year
 *   season         : "Summer" | "Monsoon" | "Winter"
 *   unitsConsumed  : kWh consumed that billing month
 *   avgTemperature : average °C for that month
 *   billAmount     : ₹ billed (slab-based approximation)
 *
 * Derived features (added by dataPrep.js — do NOT hard-code here):
 *   prevMonthUnits, rollingAvg3Month, monthSin, monthCos, seasonEncoded
 */

// Helper – assign season based on month number
function getSeason(month) {
  if (month >= 3 && month <= 6)  return 'Summer';
  if (month >= 7 && month <= 10) return 'Monsoon';
  return 'Winter';
}

// Helper – approximate Tamil Nadu slab billing (₹)
function calcBill(units) {
  if (units <= 100)  return units * 0;            // free slab
  if (units <= 200)  return (units - 100) * 1.5;
  if (units <= 500)  return 150 + (units - 200) * 3;
  return 150 + 900 + (units - 500) * 5;
}

// ---------------------------------------------------------------------------
// Raw monthly consumption data — Jan 2020 → Dec 2024 (60 records)
// Values represent realistic residential units consumed (kWh/month)
// ---------------------------------------------------------------------------
const rawMonthlyData = [
  // 2020
  { year: 2020, month: 1,  unitsConsumed: 118, avgTemperature: 25.2 },
  { year: 2020, month: 2,  unitsConsumed: 125, avgTemperature: 26.8 },
  { year: 2020, month: 3,  unitsConsumed: 198, avgTemperature: 29.5 },
  { year: 2020, month: 4,  unitsConsumed: 265, avgTemperature: 32.1 },
  { year: 2020, month: 5,  unitsConsumed: 312, avgTemperature: 34.6 },
  { year: 2020, month: 6,  unitsConsumed: 278, avgTemperature: 32.8 },
  { year: 2020, month: 7,  unitsConsumed: 195, avgTemperature: 30.2 },
  { year: 2020, month: 8,  unitsConsumed: 188, avgTemperature: 29.8 },
  { year: 2020, month: 9,  unitsConsumed: 175, avgTemperature: 29.5 },
  { year: 2020, month: 10, unitsConsumed: 162, avgTemperature: 28.9 },
  { year: 2020, month: 11, unitsConsumed: 134, avgTemperature: 26.4 },
  { year: 2020, month: 12, unitsConsumed: 120, avgTemperature: 24.8 },
  // 2021
  { year: 2021, month: 1,  unitsConsumed: 115, avgTemperature: 24.5 },
  { year: 2021, month: 2,  unitsConsumed: 122, avgTemperature: 26.1 },
  { year: 2021, month: 3,  unitsConsumed: 205, avgTemperature: 30.2 },
  { year: 2021, month: 4,  unitsConsumed: 272, avgTemperature: 33.4 },
  { year: 2021, month: 5,  unitsConsumed: 325, avgTemperature: 35.8 },
  { year: 2021, month: 6,  unitsConsumed: 290, avgTemperature: 33.5 },
  { year: 2021, month: 7,  unitsConsumed: 202, avgTemperature: 30.8 },
  { year: 2021, month: 8,  unitsConsumed: 193, avgTemperature: 30.1 },
  { year: 2021, month: 9,  unitsConsumed: 180, avgTemperature: 29.8 },
  { year: 2021, month: 10, unitsConsumed: 168, avgTemperature: 29.2 },
  { year: 2021, month: 11, unitsConsumed: 138, avgTemperature: 27.1 },
  { year: 2021, month: 12, unitsConsumed: 124, avgTemperature: 25.0 },
  // 2022
  { year: 2022, month: 1,  unitsConsumed: 119, avgTemperature: 24.8 },
  { year: 2022, month: 2,  unitsConsumed: 128, avgTemperature: 27.2 },
  { year: 2022, month: 3,  unitsConsumed: 210, avgTemperature: 30.8 },
  { year: 2022, month: 4,  unitsConsumed: 285, avgTemperature: 33.9 },
  { year: 2022, month: 5,  unitsConsumed: 338, avgTemperature: 36.2 },
  { year: 2022, month: 6,  unitsConsumed: 298, avgTemperature: 34.1 },
  { year: 2022, month: 7,  unitsConsumed: 208, avgTemperature: 31.2 },
  { year: 2022, month: 8,  unitsConsumed: 198, avgTemperature: 30.5 },
  { year: 2022, month: 9,  unitsConsumed: 185, avgTemperature: 30.0 },
  { year: 2022, month: 10, unitsConsumed: 172, avgTemperature: 29.5 },
  { year: 2022, month: 11, unitsConsumed: 142, avgTemperature: 27.5 },
  { year: 2022, month: 12, unitsConsumed: 128, avgTemperature: 25.3 },
  // 2023
  { year: 2023, month: 1,  unitsConsumed: 122, avgTemperature: 25.0 },
  { year: 2023, month: 2,  unitsConsumed: 130, avgTemperature: 27.5 },
  { year: 2023, month: 3,  unitsConsumed: 215, avgTemperature: 31.0 },
  { year: 2023, month: 4,  unitsConsumed: 292, avgTemperature: 34.2 },
  { year: 2023, month: 5,  unitsConsumed: 348, avgTemperature: 37.1 },
  { year: 2023, month: 6,  unitsConsumed: 308, avgTemperature: 34.8 },
  { year: 2023, month: 7,  unitsConsumed: 214, avgTemperature: 31.5 },
  { year: 2023, month: 8,  unitsConsumed: 204, avgTemperature: 30.8 },
  { year: 2023, month: 9,  unitsConsumed: 190, avgTemperature: 30.2 },
  { year: 2023, month: 10, unitsConsumed: 176, avgTemperature: 29.7 },
  { year: 2023, month: 11, unitsConsumed: 146, avgTemperature: 27.8 },
  { year: 2023, month: 12, unitsConsumed: 132, avgTemperature: 25.6 },
  // 2024
  { year: 2024, month: 1,  unitsConsumed: 126, avgTemperature: 25.2 },
  { year: 2024, month: 2,  unitsConsumed: 135, avgTemperature: 27.8 },
  { year: 2024, month: 3,  unitsConsumed: 222, avgTemperature: 31.5 },
  { year: 2024, month: 4,  unitsConsumed: 298, avgTemperature: 34.8 },
  { year: 2024, month: 5,  unitsConsumed: 355, avgTemperature: 37.6 },
  { year: 2024, month: 6,  unitsConsumed: 315, avgTemperature: 35.2 },
  { year: 2024, month: 7,  unitsConsumed: 220, avgTemperature: 31.8 },
  { year: 2024, month: 8,  unitsConsumed: 210, avgTemperature: 31.1 },
  { year: 2024, month: 9,  unitsConsumed: 195, avgTemperature: 30.5 },
  { year: 2024, month: 10, unitsConsumed: 180, avgTemperature: 30.0 },
  { year: 2024, month: 11, unitsConsumed: 150, avgTemperature: 28.0 },
  { year: 2024, month: 12, unitsConsumed: 136, avgTemperature: 25.8 },
];

// Attach season and bill amount to each record
export const ebBillingData = rawMonthlyData.map((d) => ({
  ...d,
  season:     getSeason(d.month),
  billAmount: parseFloat(calcBill(d.unitsConsumed).toFixed(2)),
}));

export default ebBillingData;
