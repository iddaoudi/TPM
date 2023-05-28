static const char *read_string(const char *fn)
{
    char buf[80];
    int fd;
    int rc;

    bzero(buf, sizeof(buf));

    fd = open(fn, O_RDONLY);
    if (fd < 0)
        return NULL;
    rc = read(fd, buf, sizeof(buf));
    if (rc < 0)
        buf[0] = 0;
    buf[sizeof(buf) - 1] = 0;
    close(fd);

    if ((strlen(buf) > 0) && (buf[strlen(buf) - 1] == '\n'))
        buf[strlen(buf) - 1] = 0;

    return strdup(buf);
}

int TPM_rapl_init()
{
    char fn[256];
    int rc;
    regex_t re;
    regmatch_t pm[3];
    int active_packages = 0;

    rc = regcomp(&re, "^package-([0-9]+)$", REG_EXTENDED);
    if (rc != 0)
    {
        fprintf(stderr, "regcomp() failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < MAX_PKGS; i++)
    {
        snprintf(fn, sizeof(fn), "%s/intel-rapl:%d/name", SYSFS_RAPL_DIR, i);
        if (access(fn, R_OK) == 0)
        {
            const char *s = read_string(fn);
            rc = regexec(&re, s, 3, pm, 0);
            if (rc == 0)
            {
                int pkgid = atoi(s + pm[1].rm_so);
                snprintf(fn, sizeof(fn), "%s/intel-rapl:%d/energy_uj", SYSFS_RAPL_DIR,
                         i);
                pkg_energy_uj[pkgid] = strdup(fn);

                snprintf(fn, sizeof(fn), "%s/intel-rapl:%d/max_energy_range_uj",
                         SYSFS_RAPL_DIR, i);
                pkg_energy_maxuj[pkgid] = strdup(fn);

                snprintf(fn, sizeof(fn), "%s/intel-rapl:%d/intel-rapl:%d:0/energy_uj",
                         SYSFS_RAPL_DIR, i, i);
                dram_energy_uj[pkgid] = strdup(fn);

                snprintf(fn, sizeof(fn),
                         "%s/intel-rapl:%d/intel-rapl:%d:0/max_energy_range_uj",
                         SYSFS_RAPL_DIR, i, i);
                dram_energy_maxuj[pkgid] = strdup(fn);
                active_packages++;
            }
        }
        else
        {
            pkg_energy_uj[i] = NULL;
            dram_energy_uj[i] = NULL;
            pkg_energy_maxuj[i] = NULL;
            dram_energy_maxuj[i] = NULL;
        }
    }
    return active_packages;
}

uint64_t TPM_rapl_get_maxuj(int pkgid, char *domain)
{
    char *fn;
    uint64_t ret = 0;

    if (pkgid < 0 || pkgid >= MAX_PKGS)
        return 1;

    if (strcmp(domain, "pkg") == 0)
        fn = pkg_energy_maxuj[pkgid];
    else if (strcmp(domain, "dram") == 0)
        fn = dram_energy_maxuj[pkgid];

    if (!fn)
        return 2;

    const char *s = read_string(fn);
    if (!s)
        return 3;
    ret = atoll(s);
    return ret;
}

uint64_t TPM_rapl_get_uj(int pkgid, char *domain)
{
    char *fn;
    uint64_t ret = 0;

    if (pkgid < 0 || pkgid >= MAX_PKGS)
        return 1;

    if (strcmp(domain, "pkg") == 0)
        fn = pkg_energy_uj[pkgid];
    else if (strcmp(domain, "dram") == 0)
        fn = dram_energy_uj[pkgid];

    if (!fn)
        return 2;

    const char *s = read_string(fn);
    if (!s)
        return 3;
    ret = atoll(s);
    return ret;
}