
# Parte 1
import re
import os

# T1

def leParagrafos(ficheiro:str) -> list[str]:
    """
    Reads paragraphs from a text file, ensuring each paragraph is treated as a single string.

    Parameters:
    - file_path (str): The path to the text file to be read.

    Returns:
    - list[str]: A list of paragraphs as strings, with whitespace and new lines stripped.
    """
    paragraphs = []  # Initialize an empty list to store the paragraphs
    
    with open(ficheiro, "r", encoding="utf-8-sig") as file:
        content = file.read()
        raw_paragraphs = content.split('\n\n')  # Split the content by double new lines to separate paragraphs
        for paragraph in raw_paragraphs:
            cleaned_paragraph = paragraph.strip('\n').replace('\n', ' ')  # Remove leading/trailing new lines and replace internal new lines with spaces
            if cleaned_paragraph:  # Ensure the paragraph is not just whitespace
                paragraphs.append(cleaned_paragraph)
    return paragraphs

# T2

def organizaCapitulos(paragrafos: list[str]) -> dict[str, list[str]]:
    """
    Organizes paragraphs into chapters based on Roman numeral chapter numbers.

    Args:
        paragrafos (list[str]): A list of paragraphs.

    Returns:
        dict[str, list[str]]: A dictionary where the keys are Roman numeral chapter numbers
        and the values are lists of paragraphs belonging to each chapter.
    """
    chap_dict = {}
    roman_numeral_regex = r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
    chap_num_key = None
    list_of_chapter_paragraphs = []

    for paragrafo in paragrafos[2:]:
        chap_num = re.fullmatch(roman_numeral_regex, paragrafo.upper())  # Check if the paragraph is a chapter number
        if chap_num is not None:  # If the paragraph is a chapter number
            chap_dict[chap_num.group()] = []  # Add a new key to the dictionary
            chap_num_key = chap_num.group()  # Store the current chapter number
            list_of_chapter_paragraphs = []  # Reset the list of chapter paragraphs

        elif chap_num is None:  # If the paragraph is not a chapter number
            list_of_chapter_paragraphs.append(paragrafo)  # Add the paragraph to the current chapter key
            chap_dict[chap_num_key] = list_of_chapter_paragraphs  # Update the dictionary with the new paragraph
    return chap_dict

# T3

def menorCapitulo(capitulos: dict[str, list[str]]) -> str:
    """
    Finds and returns the name of the chapter with the smallest total size of its subchapters.

    Parameters:
    - capitulos (dict[str, list[str]]): A dictionary where the keys represent chapter names and the values are lists of subchapters.

    Returns:
    - str: The name of the chapter with the smallest total size of its subchapters.
    """
    min_chap_size = float('inf')
    for key in capitulos.keys():
        if sum(map(len, capitulos[key])) < min_chap_size:
            min_chap_size = sum(map(len, capitulos[key]))
            min_chap_name = key
    return min_chap_name

def maiorDialogo(capitulos: dict[str, list[str]]) -> int:
    """
    Finds the length of the longest consecutive dialogue in a given dictionary of chapters.

    Parameters:
    - capitulos (dict[str, list[str]]): A dictionary where the keys represent chapter names and the values represent paragraphs.

    Returns:
    - int: The length of the longest consecutive dialogue.

    Example:
    >>> capitulos = {
    ...     "Chapter 1": ["-- Dialogue 1", "Paragraph 1", "-- Dialogue 2", "-- Dialogue 3", "Paragraph 2"],
    ...     "Chapter 2": ["Paragraph 3", "-- Dialogue 4", "-- Dialogue 5", "-- Dialogue 6"],
    ...     "Chapter 3": ["Paragraph 4", "Paragraph 5", "-- Dialogue 7", "-- Dialogue 8", "-- Dialogue 9"]
    ... }
    >>> maiorDialogo(capitulos)
    3
    """

    most_consecutive_dialogs = 0
    current_consecutive_dialogs = 0
    
    for key in capitulos.keys():
        for paragraph in capitulos[key]:
            if paragraph.startswith('--'):
                current_consecutive_dialogs += 1
            else:
                if current_consecutive_dialogs > most_consecutive_dialogs: # Check if the current sequence is the longest
                    most_consecutive_dialogs = current_consecutive_dialogs # Update the longest sequence
                current_consecutive_dialogs = 0
    
    # Check last sequence in case the longest dialogue is at the end of the chapters
    
    if current_consecutive_dialogs > most_consecutive_dialogs:
        most_consecutive_dialogs = current_consecutive_dialogs

    return most_consecutive_dialogs

def roman_to_int(s):
    """
    Converts a Roman numeral string to an integer.

    Args:
        s (str): The Roman numeral string to be converted.

    Returns:
        int: The integer representation of the Roman numeral.

    Examples:
        >>> roman_to_int('III')
        3
        >>> roman_to_int('IX')
        9
        >>> roman_to_int('LVIII')
        58
    """
    roman_numerals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    int_val = 0
    for i in range(len(s)):
        if i > 0 and roman_numerals[s[i]] > roman_numerals[s[i - 1]]:
            int_val += roman_numerals[s[i]] - 2 * roman_numerals[s[i - 1]]
        else:
            int_val += roman_numerals[s[i]]
    return int_val

def mencoesPersonagens(capitulos: dict[str, list[str]], personagens: set[str]) -> list[tuple[str, float]]:
    """
    Counts the number of mentions of each character in each chapter and returns a sorted list of tuples.

    Args:
        capitulos (dict[str, list[str]]): A dictionary containing chapters as keys and a list of paragraphs as values.
        personagens (set[str]): A set of characters to count mentions for.

    Returns:
        list[tuple[str, float]]: A sorted list of tuples, where each tuple contains a chapter and the number of mentions of the characters in that chapter.
    """

    mencoes = {}
    for capitulo, paragrafos in capitulos.items():
        for paragrafo in paragrafos:
            lista_personagens = [personagem in paragrafo for personagem in personagens] # Check if each character is mentioned in the paragraph
            condition = all(lista_personagens) # Check if all characters are mentioned in the paragraph
            if condition:                      
                mencoes[capitulo] = mencoes.get(capitulo, 0) + 1
            else:
                mencoes[capitulo] = mencoes.get(capitulo, 0)
    mencoes_list = list(mencoes.items()) # Convert the dictionary to a list of tuples
    mencoes_list.sort(key=lambda x: (-x[1], roman_to_int(x[0]))) # Sort the list by the number of mentions and then by the chapter number
    return mencoes_list

def ohJacinto(capitulos:dict[str,list[str]]) -> set[str]:
    """
    Extracts monologues from a dictionary of chapters and paragraphs.

    Args:
        capitulos (dict[str, list[str]]): A dictionary containing chapters as keys and lists of paragraphs as values.

    Returns:
        set[str]: A set of monologues that meet the specified criteria.
    """
    # Função para verificar se "Jacintho" está fora de parênteses e acompanhado por caracteres permitidos
    def validar_frase(frase):
        profundidade_parenteses = 0
        for i, char in enumerate(frase):
            if char == '(':
                profundidade_parenteses += 1
            elif char == ')':
                profundidade_parenteses -= 1
            elif profundidade_parenteses == 0 and frase[i:i+8] == "Jacintho":
                # Verifica se "Jacintho" está seguido e/ou precedido por caracteres permitidos
                if i + 8 == len(frase) or frase[i+8] in ", !?":
                    if i == 0 or frase[i-1] in " ,":
                        return True
        return False
    
    monologues = set()
    for _, paragrafos in capitulos.items():
        for paragrafo in paragrafos:
            if paragrafo.startswith('--'):
                # Primeiro passo: Captura de candidatos a frases
                candidatos = re.findall(r'[^!?]*?[!?]', paragrafo)    
                for frase in candidatos:
                    # Segundo passo: Validação dos candidatos
                    if validar_frase(frase):
                        monologues.add(paragrafo.strip()) 
                        # Adiciona a sentença que cumpre os critérios ao conjunto de monólogos
                        # monologues.add(sentence.strip())
    
    # this is a log to help debug the function           
    # file_path = 'dump\monologues_res.py'
    # #if not os.path.exists(file_path):
    # with open(file_path, 'w', encoding="utf-8") as file:
    #     file.write("ohjacinto_res = {")
    #     monologues_list = list(monologues)
    #     for monologue in monologues_list[:-1]:
    #         file.write('"' + str(monologue) + '"' + ',')
    #     file.write('"' + str(monologues_list[-1]) + '"')
    #     file.write("}")
    return monologues

# Parte 2

# T4

# the nucleotide complement of a DNA nucleotide in its bonded DNA strand
nucleotidePairs = {'A':'T','G':'C','Y':'R','W':'W','S':'S','K':'M','D':'H','V':'B','X':'X','N':'N'}
for k,v in list(nucleotidePairs.items()): nucleotidePairs[v] = k

def nucleotidePair(c):
    return nucleotidePairs[c]

def leDNA(ficheiro: str) -> tuple[str, str]:
    """
    Reads a DNA sequence from a file and returns it along with its complement sequence.

    Parameters:
    - ficheiro (str): The path to the file containing the DNA sequence. The first line is ignored as it is assumed to be a header.

    Returns:
    - tuple[str, str]: A tuple containing the DNA sequence and its complement sequence.
    """
    # Open the file and read the DNA sequence
    with open(ficheiro, 'r') as file:
        file.readline()  # Skip the first line (header)
        dna_sequence = ''.join(line.strip() for line in file)  # Read and concatenate the remaining lines to form the DNA sequence
        complement_sequence = ''.join(nucleotidePair(nucleotide) for nucleotide in dna_sequence)  # Generate the complement sequence
        return dna_sequence, complement_sequence

# T5

def encontraProteinas(code: str, dna: str) -> list[tuple[int, int, str]]:
    """
    Finds proteins encoded in a given DNA sequence based on codon mappings.

    Parameters:
    - code (str): Path to the file containing codon to amino acid mappings.
    - dna (str): The DNA sequence to be searched for proteins.

    Returns:
    - list[tuple[int, int, str]]: A list of tuples each representing a found protein with start position, end position, and the amino acid sequence.
    """
    # Read codon mappings from the provided file
    with open(code, 'r') as file:
        lines = file.readlines()
    code_dict = {line.split('=')[0].strip(): line.split('=')[1].strip() for line in lines}

    # Maps each codon to its corresponding amino acid
    codon_to_aa = {code_dict['Base1'][i] + code_dict['Base2'][i] + code_dict['Base3'][i]: code_dict['AAs'][i] for i in range(64)}
    codon_to_start = {code_dict['Base1'][i] + code_dict['Base2'][i] + code_dict['Base3'][i] for i in range(64) if code_dict['Starts'][i] == 'M'}

    proteins = []  # List to store the proteins found in the DNA sequence

    def traduz_seq(seq: str) -> list[tuple[int, int, str]]:
        """
        Translates a DNA sequence into amino acids and identifies protein sequences.

        Parameters:
        - seq (str): The DNA sequence to translate and search for proteins.

        Returns:
        - list[tuple[int, int, str]]: A list of identified proteins with their start and end positions and amino acid sequences.
        """
        i = 0
        while i < len(seq) - 2:
            codon = seq[i:i+3]

            if codon in codon_to_start:  # Check if the current codon is a start codon
                start_pos = i + 1
                proteina = codon_to_aa[codon]
                i += 3  # Jump to the next codon
                while i <= len(seq) - 3:
                    codon = seq[i:i+3]
                    if codon_to_aa[codon] == '*':  # Check if the current codon is a stop codon
                        end_pos = i + 3  # Store the end position of the protein
                        if proteina:
                            proteins.append((start_pos, end_pos, proteina))
                        break
                    else:
                        proteina += codon_to_aa[codon]
                        i += 3
                i = start_pos  # Move the index to the next codon after the stop codon
            else:
                i += 1
        return proteins

    return traduz_seq(dna)

def orfFinder(code:str,dna:tuple[str,str]) -> list[tuple[int,int,str,str]]:
    l = len(dna[0])
    res = []
    for i,j,seq in encontraProteinas(code,dna[0]):
        res.append((i,j,seq,"+"))
    for i,j,seq in encontraProteinas(code,dna[1][::-1]):
        res.append((l-i+1, l-j+1, seq,"-"))
    return res

# T6

def intergenicRegions(dna: tuple[str, str], cds: str) -> str:
    """
    Identifies intergenic regions between coding sequences (CDS) within a given DNA sequence.
    
    Parameters:
    - dna (tuple[str, str]): A tuple containing two DNA sequences, representing the forward and reverse strands.
    - cds (str): The path to a file containing CDS information in a specific format.
    
    Returns:
    - str: Formatted string listing intergenic regions with their locations and the names of adjacent CDS.
    """
    
    import re  # Regular expression module for parsing CDS information
    
    def extract_locations_and_names(cds_line: str) -> tuple[int, int, bool, str]:
        """
        Extracts the start and end positions, orientation, and name of a CDS from a given line of text.
        
        Parameters:
        - cds_line (str): A line from the CDS file containing location and name information.
        
        Returns:
        - tuple[int, int, bool, str]: Start position, end position, orientation (True if complement), and name of the CDS.
        """
        # Regex to extract location and orientation
        location_match = re.search(r'\[location=(complement\()?([<>]?\d+)\.\.([<>]?\d+)(\))?]', cds_line)
        # Regex to extract the CDS name
        name_match = re.search(r'>lcl\|([^ ]+)', cds_line)
        
        if location_match and name_match:
            # Extracting start and end positions, accounting for '<' and '>' symbols
            start = int(location_match.group(2).lstrip('<>'))
            end = int(location_match.group(3).lstrip('<>'))
            # Determining if the CDS is on the complement strand
            is_complement = location_match.group(1) is not None
            # Extracting the name of the CDS
            name = name_match.group(1)
            return start, end, is_complement, name
        return (0, 0, False, "")  # Returns a default tuple if no match is found
    
    cds_info = []  # List to store extracted CDS information
    with open(cds, 'r') as file:
        for line in file:
            if line.startswith('>'):  # Identifies lines containing CDS information
                cds_info.append(extract_locations_and_names(line))
    
    # Sort the CDS information by start location to process in order
    cds_info.sort(key=lambda x: x[0])

    intergenic_regions = []  # List to store identified intergenic regions
    i = 0
    # Iterate through the CDS information to find intergenic regions
    while i < len(cds_info) - 1:
        # Current and next CDS information
        start_current, end_current, is_complement_current, name_current = cds_info[i]
        start_next, end_next, is_complement_next, name_next = cds_info[i + 1]
        
        # Handle overlapping CDS regions
        if end_current > end_next:
            j = i + 1
            while j < len(cds_info) - 1:
                start_next, end_next, is_complement_next, name_next = cds_info[j + 1]
                
                # Skip if strands differ
                if is_complement_current != is_complement_next:
                    j += 1
                    continue
                
                # Check for non-overlapping region
                if end_current < start_next - 1:
                    intergenic_regions.append((end_current + 1, start_next - 1, is_complement_current, name_current, name_next))
                    break
                j += 1
            i += 1
            continue
        
        # Handle change of strand
        if is_complement_current != is_complement_next:
            j = i + 1
            while j < len(cds_info) - 1:
                start_next, end_next, is_complement_next, name_next = cds_info[j + 1]
                
                if is_complement_current != is_complement_next:
                    j += 1
                    continue
                else:
                    if end_current < start_next - 1:
                        intergenic_regions.append((end_current + 1, start_next - 1, is_complement_current, name_current, name_next))
                        break
                j += 1  
            i += 1
            continue

        # Identify and store non-overlapping intergenic regions
        if end_current < start_next - 1:
            intergenic_regions.append((end_current + 1, start_next - 1, is_complement_current, name_current, name_next))
        
        i += 1

    # Sort intergenic regions by their end position
    intergenic_regions.sort(key=lambda x: x[1])
    output = ""  # String to store the final output
    # Formatting the output
    for start, end, is_complement, name_current, name_next in intergenic_regions:
        direction = "-" if is_complement else "+"
        # Extracting the DNA segment from the appropriate strand
        dna_segment = dna[1][start:end+1] if is_complement else dna[0][start:end+1]
        # Formatting the output for each intergenic region
        if is_complement:
            output += f">lcl|{name_current}..lcl|{name_next} [location=complement({start}..{end})] {direction}\n{dna_segment}\n"
        else:
            output += f">lcl|{name_current}..lcl|{name_next} [location={start}..{end}] {direction}\n{dna_segment}\n"
    
    return output
