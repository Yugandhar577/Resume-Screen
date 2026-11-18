# India salary comparison

The local comparison uses a small attributed set of published numeric facts. It does
not redistribute reports or claim coverage of every Indian occupation or city.

## Included references

- [Syncwell India IT salary report 2025](https://syncwell.co.in/syncwell-infotech-2025-it-software-salary-survey-report/):
  eight role/experience CTC ranges. Recruiter observations, not a representative
  national survey; sample sizes and percentile distributions are unavailable.
- [foundit October 2025 India report](https://www.foundit.in/career-advice/wp-content/uploads/2025/11/Salary-Trends-Report.pdf):
  four industry categories across five experience brackets. These are published
  average-range references, not individual percentiles. Compensation basis is not
  specified, so the app shows them without declaring a salary mismatch.

Source dates, geography, method, pay basis and links appear beside results. No city
adjustment or inflation estimate is applied. Historical figures are not described as
current quotations. At the overlapping 15-year industry boundary, the explicitly
selected source bracket determines the comparison. Industry brackets use completed
years; role-specific ranges retain the confirmed fractional experience.

## What the comparison checks

- INR amounts, annual/monthly periods, LPA, lakh and crore scaling.
- A single bounded offer range, ordered positive amounts and matching currency.
- Confirmed experience within the selected reference bracket.
- Base pay versus CTC and contradictions between the offer text and confirmation.
- Multiple compensation figures, mixed currencies, open-ended or conflicting units
  produce an explicit clarification state instead of a fabricated comparison.

Below/overlapping/above means only the extracted range's position relative to the
selected source. It is not an underpayment finding or a personal salary prediction.
Job equivalence, proficiency, employer type and benefits still need interpretation.

## Use another permitted benchmark

In Results → India Salary Comparison, select **Use a local benchmark JSON**. The
file stays in page memory and is sent only to the local comparison API. No arbitrary
source URL is fetched, no provider key is needed, and no benchmark is silently saved.
Values supplied by the user are explicitly identified as not independently verified.

The file must contain one object with these fields:

| Field | Required value |
| --- | --- |
| title, role | Source title and occupation/category |
| url | Valid HTTP/HTTPS source URL |
| as_of | Actual source date: YYYY, YYYY-MM or YYYY-MM-DD; not a future date |
| country, currency | India, INR |
| pay_basis | ctc or base |
| geography, method | Actual coverage and how the source produced its figures |
| license | Permission/license basis for your use of the values |
| experience_min, experience_max | Ordered experience range in years, between 0 and 60 |
| annual_low, annual_high | Ordered positive annual rupee amounts, not LPA units |
| id | Optional local identifier |

Use real source values; an illustrative fixture is not a market benchmark. Unknown
fields, invalid dates, nonfinite numbers and reversed ranges are rejected. Source
permission and factual accuracy of a user-supplied file are attestations, not identity
or source verification performed by the app.
