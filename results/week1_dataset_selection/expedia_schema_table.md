# Expedia `train.csv` schema table (37,670,293 rows, 24 columns)

Coverage = share of non-null values. `all` = whole file; `bookings` = the
`is_booking == 1` subset the benchmark trains on.

| No. | Group | Field | Description | Coverage (all) | Coverage (bookings) | Distinct | Range | Sample values |
|----:|-------|-------|-------------|---------------:|--------------------:|---------:|-------|---------------|
| 1 | Interaction / time | `cnt` | Number of similar events in the same user session context | 100.00% | 100.00% | 104 | 1 .. 269 | 3, 1, 2 |
| 2 | Interaction / time | `date_time` | Timestamp of the search event (second resolution); the temporal-split key | 100.00% | 100.00% | n/a | 2013-01-07 .. 2014-12-31 | 2014-08-11 07:46:59, 2014-08-11 08:22:12, 2014-08-11 08:24:33 |
| 3 | Interaction / time | `is_booking` | Event label: 1 = booking, 0 = click | 100.00% | 100.00% | 2 | 0 .. 1 | 0, 1 |
| 4 | User identity | `user_id` | Anonymised traveller id; the only personalisation key in the file | 100.00% | 100.00% | 1,198,786 | 0 .. 1,198,785 | 12, 93, 501 |
| 5 | Trip context | `is_package` | 1 = booked as part of a package (flight/car bundled) | 100.00% | 100.00% | 2 | 0 .. 1 | 1, 0 |
| 6 | Trip context | `srch_adults_cnt` | Adults in the room request | 100.00% | 100.00% | 10 | 0 .. 9 | 2, 1, 3 |
| 7 | Trip context | `srch_children_cnt` | Children in the room request | 100.00% | 100.00% | 10 | 0 .. 9 | 0, 2, 3 |
| 8 | Trip context | `srch_ci` | Check-in date requested; drives the seasonality slice | 99.88% | 100.00% | 1,269 | 2012-02-15 .. 2558-03-15 | 2014-08-27, 2014-08-29, 2014-11-23 |
| 9 | Trip context | `srch_co` | Check-out date requested; with srch_ci gives stay length | 99.88% | 100.00% | 1,262 | 2012-09-04 .. 2558-03-16 | 2014-08-31, 2014-09-02, 2014-11-28 |
| 10 | Trip context | `srch_rm_cnt` | Rooms requested | 100.00% | 100.00% | 9 | 0 .. 8 | 1, 2, 3 |
| 11 | Search / destination | `srch_destination_id` | Id of the destination the user searched for | 100.00% | 100.00% | 59,455 | 0 .. 65,107 | 8250, 14984, 8267 |
| 12 | Search / destination | `srch_destination_type_id` | Type of the searched destination (city, region, landmark, ...) | 100.00% | 100.00% | 10 | 0 .. 9 | 1, 6, 4 |
| 13 | Hotel / target | `hotel_cluster` | TARGET: one of 100 anonymised hotel clusters (proxy for room category / package) | 100.00% | 100.00% | 100 | 0 .. 99 | 1, 80, 21 |
| 14 | Hotel / target | `hotel_continent` | Continent the hotel is in | 100.00% | 100.00% | 7 | 0 .. 6 | 2, 0, 3 |
| 15 | Hotel / target | `hotel_country` | Country the hotel is in | 100.00% | 100.00% | 213 | 0 .. 212 | 50, 185, 151 |
| 16 | Hotel / target | `hotel_market` | Local market (sub-country travel market) of the hotel | 100.00% | 100.00% | 2,118 | 0 .. 2,117 | 628, 1457, 675 |
| 17 | User geography | `orig_destination_distance` | Physical distance between user city and hotel, in miles; blank when it could not be computed | 64.10% | 66.17% | 8,495,289 | 0.01 .. 12,407.90 | 2234.26416015625, 913.1931762695312, 913.6259155273438 |
| 18 | User geography | `user_location_city` | City the user is browsing from | 100.00% | 100.00% | 50,447 | 0 .. 56,508 | 48862, 35390, 10067 |
| 19 | User geography | `user_location_country` | Country the user is browsing from | 100.00% | 100.00% | 237 | 0 .. 239 | 66, 195, 69 |
| 20 | User geography | `user_location_region` | Region within the user country | 100.00% | 100.00% | 1,008 | 0 .. 1,027 | 348, 442, 189 |
| 21 | Channel / platform | `channel` | Marketing channel the user arrived through | 100.00% | 100.00% | 11 | 0 .. 10 | 9, 3, 2 |
| 22 | Channel / platform | `is_mobile` | 1 = event came from a mobile device | 100.00% | 100.00% | 2 | 0 .. 1 | 0, 1 |
| 23 | Channel / platform | `posa_continent` | Continent of the point of sale | 100.00% | 100.00% | 5 | 0 .. 4 | 3, 4, 1 |
| 24 | Channel / platform | `site_name` | Expedia point-of-sale website (e.g. Expedia.com vs a locale site) | 100.00% | 100.00% | 45 | 2 .. 53 | 2, 30, 37 |
