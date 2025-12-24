log = {
        'helpdesk': {'event_attribute': ['activity', 'resource', 'timesincecasestart','servicelevel','servicetype','workgroup','product','customer'], 'trace_attribute': ['supportsection','responsiblesection'],
                     'event_template': '{{resource}} performed {{activity}} {{timesincecasestart}} seconds ago from the beginning of the trace. {{workgroup}} managed the request for the {{product}} of {{customer}} with service {{servicetype}} of level {{servicelevel}}.',
                     'trace_template': 'Section {{supportsection}} led by {{responsiblesection}}', 'target':'activity'},

        'bpic2020': {'event_attribute': ['activity','resource','timesincecasestart', 'Role'], 'trace_attribute': ['Org','Project','Task'],
                     'event_template': '{{resource}} with role {{Role}} performed {{activity}} {{timesincecasestart}} seconds ago from the beginning of the trace.',
                     'trace_template': '{{Org}} managed the {{Project}} for {{Task}}.', 'target':'activity'},

        'bpic2017_o': {'event_attribute': ['activity', 'resource', 'timesincecasestart', 'action'], 'trace_attribute': ["MonthlyCost", "CreditScore", "FirstWithdrawalAmount", "OfferedAmount","NumberOfTerms"],
                       'event_template': '{{resource}} performed {{activity}} with status {{action}} {{timesincecasestart}} seconds ago from the beginning of the trace.',
                       'trace_template': 'The MonthlyCost {{MonthlyCost}} for the loan, determined based on the score {{CreditScore}}, calculated considering the FirstWithdrawalAmount {{FirstWithdrawalAmount}}, the OfferedAmount {{OfferedAmount}}, and the NumberOfTerms {{NumberOfTerms}}.','target': 'activity'},

        'BPIC12_W':{
                'event_attribute': ['activity', 'resource','timesincecasestart'],'trace_attribute': ["case_start"],
                'event_template': '{{resource}} performed {{activity}} {{timesincecasestart}} seconds ago from the beginning of the trace.',
                'trace_template': 'The trace starts at {{case_start}}', 'target':'activity'},

        'BPIC2013p':{
            'event_attribute': ['activity', 'resource','timesincecasestart', 'resourcecountry', 'lifecycletransition'],'trace_attribute': ['product', 'group', 'organizationinvolved', "organizationcountry", 'role', 'impact', 'product'],
            'event_template': '{{resource}} performed {{activity}} {{timesincecasestart}} seconds ago from the beginning of the trace.from {{resourcecountry}} with status {{lifecycletransition}}.',
            'trace_template': 'This event was carried out by the organization {{organizationinvolved}} from {{organizationcountry}}.The issue had an impact of {{impact}} and concerned the product {{product}}.','target':'activity'},

        'Receipt':{
                'event_attribute': ['activity', 'resource','timesincecasestart','group', 'instance'],'trace_attribute': ['channel', 'department', 'casegroup', 'responsible' ],
                'event_template': '{{resource}} performed {{activity}} {{timesincecasestart}} seconds ago from the beginning of the trace.within group {{group}} and instance {{instance}}.',
                'trace_template': 'The application was submitted via the {{channel}} channel and is handled by the {{department}} department, with the {{casegroup}} workgroup and {{responsible}} responsible person in charge.', 'target':'activity'},
        }