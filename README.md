This projects aims to improve source and sink classification in Android Apps. 

Sources and Sinks (SS) are used in data flow analysis tools to constrain analysis to only 
those methods that produce sensitive information (source) and those that transmit sensitive 
information outside of an application (sink). Previous SS classification tools have only looked 
at a limited number of features (~100) obtained from static analysis and simple boolean keyword 
features extracted from a method's signature. This tool explores additionally using a much larger 
number of features extracted from each Android API method's documentation.
