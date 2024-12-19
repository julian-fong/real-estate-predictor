## Some Notes on the raw data


Fields:
- Zip Code:

The first three digits indicate the forwarding station, or postal district, which usually corresponds to a province or territory. The first letter of the postal code indicates the province or territory. The next digit indicates if the property is urban or rural, and the third digit recognizes a subdivision. The last three digits break down the local delivery unit.


    - Somes in ### ### format and also ###### format, 
    - Can also contain bad values that can have a missing values eg: "A1A A2"

- listDate and soldDate 
    - in "2024-01-01T00:00:00.000Z" formati
    - potentially can contain the "None" string inside

- Ammenities
    - Unique values so far: 
        ['School',
        'Library',
        'Public Transit',
        'Clear View',
        'Park',
        'Golf',
        'Arts Centre',
        'Hospital',
        'Place Of Worship',
        'Cul De Sac',
        'Ravine',
        'Fenced Yard',
        'Rec Centre',
        'School Bus Route',
        'Beach',
        'Lake/Pond',
        'Skiing',
        'Waterfront',
        'Electric Car Charger',
        'Grnbelt/Conserv',
        'Campground',
        'Marina',
        'Other',
        'Level',
        'Wooded/Treed',
        'Lake Access',
        'River/Stream',
        'Island',
        'Terraced',
        'Rolling',
        'Part Cleared',
        'Sloping',
        'Lake Backlot',
        'Tiled',
        'Lake/Pond/River',
        'Tiled/Drainage',
        'Electric Car Charg']

- Condo Ammenities
    - Unique values so far:
        ['Concierge',
        'Exercise Room',
        'Party/Meeting Room',
        'Recreation Room',
        'Rooftop Deck/Garden',
        'Visitor Parking',
        'Gym',
        'Security Guard',
        'Guest Suites',
        'Outdoor Pool',
        'Sauna',
        'Security System',
        'Car Wash',
        'Games Room',
        'Indoor Pool',
        'Media Room',
        'Bbqs Allowed',
        'Bus Ctr (Wifi Bldg)',
        'Bike Storage',
        'Tennis Court',
        'Squash/Racquet Court',
        'Lap Pool',
        'Satellite Dish']